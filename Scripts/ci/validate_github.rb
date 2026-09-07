#!/usr/bin/env ruby
# Validate repository GitHub configuration.
require 'yaml'

REQUIRED_JOBS = %w[test lint repository-checks consumer platforms].freeze

def assert_policy(condition, message)
  raise ArgumentError, message unless condition
end

def check_duplicate_keys(node)
  if node.is_a?(Psych::Nodes::Mapping)
    keys = node.children.each_slice(2).map(&:first).map do |key|
      assert_policy(key.is_a?(Psych::Nodes::Scalar), 'YAML mapping keys must be scalars')
      key.value
    end
    assert_policy(keys.uniq.length == keys.length, 'duplicate YAML mapping key')
  end
  Array(node.children).each { |child| check_duplicate_keys(child) } if node.respond_to?(:children)
end

def parse_yaml(text)
  tree = Psych.parse_stream(text)
  assert_policy(tree.children.length == 1, 'expected exactly one YAML document')
  check_duplicate_keys(tree)
  value = YAML.safe_load(text, permitted_classes: [], permitted_symbols: [], aliases: false)
  assert_policy(value.is_a?(Hash), 'expected a YAML mapping')
  value
end

def validate_permissions(permissions, codeql: false)
  assert_policy(permissions.is_a?(Hash) && permissions['contents'] == 'read',
                'explicit permissions must include contents: read')
  permissions.each do |scope, level|
    allowed = %w[read none].include?(level) ||
              (codeql && scope == 'security-events' && level == 'write')
    assert_policy(allowed, "excess permission #{scope}: #{level}")
  end
end

def validate_action(reference)
  assert_policy(reference.is_a?(String) &&
                reference.match?(%r{\A[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-fA-F]{40}\z}),
                "action must use a full 40-hex commit pin: #{reference.inspect}")
end

def validate_workflow(workflow)
  validate_permissions(workflow['permissions'])
  jobs = workflow['jobs']
  assert_policy(jobs.is_a?(Hash) && !jobs.empty?, 'workflow requires jobs')
  jobs.each do |id, job|
    assert_policy(job.is_a?(Hash), "#{id}: job must be a mapping")
    steps = job.fetch('steps', [])
    assert_policy(steps.is_a?(Array), "#{id}: steps must be an array")
    references = steps.map { |step| step['uses'] if step.is_a?(Hash) }.compact
    codeql = references.any? { |ref| ref.is_a?(String) && ref.start_with?('github/codeql-action/') }
    validate_permissions(job['permissions'], codeql: codeql) if job.key?('permissions')
    validate_action(job['uses']) if job.key?('uses')
    steps.each do |step|
      assert_policy(step.is_a?(Hash), "#{id}: step must be a mapping")
      next unless step.key?('uses')

      validate_action(step['uses'])
      next unless step['uses'].split('@').first.downcase == 'actions/checkout'

      options = step['with']
      assert_policy(options.is_a?(Hash) && options['persist-credentials'] == false,
                    "#{id}: checkout requires persist-credentials: false")
    end
  end
end

def validate_ci(workflow)
  # Psych uses YAML 1.1, where the unquoted GitHub Actions key `on` parses as true.
  triggers = workflow['on'] || workflow[true]
  events = triggers.is_a?(Hash) ? triggers.keys : Array(triggers)
  assert_policy(events.include?('pull_request'), 'CI must run on pull_request')
  if triggers.is_a?(Hash)
    triggers.each_value do |options|
      next unless options.is_a?(Hash)

      assert_policy((options.keys & %w[paths paths-ignore]).empty?, 'CI must not filter paths')
    end
  end
  jobs = workflow.fetch('jobs')
  aggregates = jobs.values.select { |job| job['name'] == 'CI Required' }
  assert_policy(aggregates.length == 1, 'CI requires exactly one CI Required aggregate')
  aggregate = aggregates.first
  condition = aggregate['if'].to_s.strip.sub(/\A\$\{\{\s*/, '').sub(/\s*\}\}\z/, '').strip
  assert_policy(condition.match?(/\Aalways\s*\(\s*\)\z/), 'CI Required must run unconditionally with always()')
  needs = Array(aggregate['needs'])
  assert_policy(needs.sort == REQUIRED_JOBS.sort, 'CI Required needs must contain exactly all required jobs')
  assert_policy((REQUIRED_JOBS - jobs.keys).empty?, 'CI is missing required job definitions')
end

root = File.expand_path(ARGV.fetch(0, '../..'), ARGV.empty? ? __dir__ : Dir.pwd)
github = File.join(root, '.github')
errors = []
workflows = {}
files = Dir.glob(File.join(github, '**', '*.{yaml,yml}')) +
        Dir.glob(File.join(github, 'ISSUE_TEMPLATE', '**', '*.md'))
files.sort.each do |path|
  begin
    text = File.read(path)
    if File.extname(path) == '.md'
      match = text.match(/\A---\r?\n(.*?)^---\s*\r?$/m)
      assert_policy(match, 'issue template requires closed YAML frontmatter')
      text = match[1]
    end
    assert_policy(!text.match?(/\byourusername\b/i), 'replace placeholder yourusername')
    data = parse_yaml(text)
    if File.dirname(path) == File.join(github, 'workflows')
      validate_workflow(data)
      workflows[File.basename(path)] = data
    end
  rescue Psych::Exception, ArgumentError => error
    errors << "#{path.delete_prefix(root + '/')}: #{error.message}"
  end
end
begin
  owners = File.read(File.join(github, 'CODEOWNERS'))
  assert_policy(!owners.match?(/\byourusername\b/i), 'CODEOWNERS contains placeholder yourusername')
  ci = workflows['ci.yml'] || workflows['ci.yaml']
  assert_policy(ci, 'missing or invalid CI workflow')
  validate_ci(ci)
rescue Errno::ENOENT, ArgumentError => error
  errors << ".github: #{error.message}"
end
unless errors.empty?
  warn errors.join("\n")
  exit 1
end
puts "Validated #{files.length} GitHub YAML/template files and workflow policies."
