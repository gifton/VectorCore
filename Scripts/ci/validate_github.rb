#!/usr/bin/env ruby
# Validate repository GitHub configuration.
require 'yaml'
require 'json'
require 'digest'

REQUIRED_JOBS = %w[test lint repository-checks consumer platforms benchmark-contracts sanitizers].freeze

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
  assert_policy(!aggregate.key?('continue-on-error'), 'CI Required cannot ignore job failure')
  assert_policy(aggregate.fetch('steps', []).none? { |step| step.key?('continue-on-error') }, 'CI Required steps cannot ignore failure')
  condition = aggregate['if'].to_s.strip.sub(/\A\$\{\{\s*/, '').sub(/\s*\}\}\z/, '').strip
  assert_policy(condition.match?(/\Aalways\s*\(\s*\)\z/), 'CI Required must run unconditionally with always()')
  needs = Array(aggregate['needs'])
  assert_policy(needs.sort == REQUIRED_JOBS.sort, 'CI Required needs must contain exactly all required jobs')
  assert_policy((REQUIRED_JOBS - jobs.keys).empty?, 'CI is missing required job definitions')
end

# Deliberately narrow current workflow grammar: these checks do not prove arbitrary shell execution.
def command_lines(job)
  assert_policy(!job.key?('if') && !job.key?('continue-on-error'), 'required jobs cannot be disabled or bypassed')
  job.fetch('steps', []).flat_map do |step|
    assert_policy(!step.key?('continue-on-error'), 'required steps cannot ignore errors')
    next [] unless step.key?('run')

    assert_policy(!step.key?('if'), 'required commands cannot be conditional')
    step['run'].to_s.gsub(/\\
\s*/, ' ').lines.map { |line| line.strip.gsub(/[ \t]+/, ' ') }.reject(&:empty?)
  end
end

def require_command(lines, pattern, label)
  assert_policy(lines.any? { |line| pattern.match?(line) }, "missing executable command: #{label}")
end

def validate_sanitizer_job(job)
  assert_policy(job.dig('strategy', 'matrix', 'sanitizer') == %w[address thread], 'sanitizers require address/thread matrix')
  lines = command_lines(job)
  require_command(lines, %r{\Apython3 Scripts/ci/check_sanitizers\.py --sanitizer "\$SANITIZER" --scratch-root "\$RUNNER_TEMP/sanitizers" --output "sanitizer-results/\$SANITIZER" --jobs [1-6]\z}, 'shared sanitizer driver')
  step = job['steps'].find { |entry| entry['run'].to_s.start_with?('python3 Scripts/ci/check_sanitizers.py ') }
  assert_policy(step.dig('env', 'SANITIZER') == '${{ matrix.sanitizer }}', 'sanitizer mode must come from matrix')
end

def validate_executable_contracts(workflow, workflows, root)
  jobs = workflow['jobs']
  [workflow, workflows.fetch('sanitizers.yml')].each do |entry|
    assert_policy(entry.dig('env', 'VECTORCORE_TEST_EXTENDED') == '0' && entry.dig('env', 'VECTORCORE_STRICT_PERF') == '0', 'normal CI requires explicit extended/strict opt-outs')
  end
  %w[Check\ memory\ pool\ under\ executor\ saturation Run\ all\ normally\ enabled\ tests Execute\ strict\ performance\ opt-out\ contracts].each do |name|
    step = jobs.fetch('test').fetch('steps').find { |entry| entry['name'] == name }
    assert_policy(step && step.dig('env', 'CONFIGURATION') == '${{ matrix.configuration }}', 'test command must use matrix configuration')
  end
  lines = REQUIRED_JOBS.to_h { |id| [id, command_lines(jobs.fetch(id))] }
  assert_policy(jobs.dig('test', 'strategy', 'matrix', 'configuration') == %w[debug release], 'tests require Debug and Release')
  assert_policy(jobs.dig('test', 'strategy', 'matrix', 'xcode') == ['16.2', '26.3'], 'preserve both test toolchains')
  require_command(lines['test'], %r{\Aswift test -c "\$CONFIGURATION" --xunit-output test-results/results\.xml 2>&1 \| tee test-results/output\.log\z}, 'normally enabled Swift tests')
  assert_policy(lines['test'].include?('set -o pipefail'), 'Swift test pipeline must propagate failure')
  require_command(lines['test'], %r{\Apython3 Scripts/ci/check_memory_pool\.py --configuration "\$CONFIGURATION"\z}, 'executor saturation check')
  require_command(lines['test'], %r{\Apython3 Scripts/ci/check_performance_gates\.py --configuration "\$CONFIGURATION" --scratch-root "\$RUNNER_TEMP/performance-gates" --output "test-results/performance-gates-\$CONFIGURATION" --jobs [1-6]\z}, 'actual strict performance process contracts')
  require_command(lines['repository-checks'], %r{\Apython3 -B -m unittest discover -s Scripts/ci/tests -v\z}, 'Python policy suite')
  require_command(lines['repository-checks'], %r{\Aruby Scripts/ci/validate_github\.rb\z}, 'GitHub policy')
  require_command(lines['lint'], %r{\Abash Scripts/ci/install_swiftlint\.sh "\$RUNNER_TEMP/swiftlint"\z}, 'pinned lint install')
  require_command(lines['lint'], %r{\A"\$RUNNER_TEMP/swiftlint/swiftlint" lint --config \.swiftlint\.yml --reporter github-actions-logging\z}, 'actual lint')
  require_command(lines['consumer'], %r{\Apython3 Scripts/ci/consumer_smoke\.py\z}, 'public documentation consumers')
  require_command(lines['platforms'], %r{\Axcodebuild -scheme VectorCore -configuration Release .*-destination "generic/platform=\$platform" .*CODE_SIGNING_ALLOWED=NO build -quiet\z}, 'Apple platform library compilation')
  require_command(lines['platforms'], %r{\Aswift build --build-tests --triple x86_64-apple-macosx14\.0 --scratch-path "\$RUNNER_TEMP/intel-test-build"\z}, 'Intel library/test compilation')
  sdk_names = command_lines(jobs.fetch('sdk-names', {}))
  require_command(sdk_names, %r{\Apython3 Scripts/ci/check_sdk_name_collisions\.py\z}, 'SDK type-name collision check (non-required sdk-names job)')
  required_needs = Array(jobs.values.find { |job| job['name'] == 'CI Required' }&.fetch('needs', nil))
  assert_policy(!required_needs.include?('sdk-names'), 'sdk-names is deliberately outside the CI Required needs')
  bench = lines['benchmark-contracts']
  require_command(bench, %r{\Aswift build -c release --product vectorcore-bench --scratch-path "\$RUNNER_TEMP/benchmark-contract-build"\z}, 'actual Release benchmark product')
  require_command(bench, %r{\ABENCH_BIN="\$\(swift build -c release --show-bin-path --scratch-path "\$RUNNER_TEMP/benchmark-contract-build"\)/vectorcore-bench"\z}, 'actual built benchmark bin path')
  require_command(bench, %r{\Apython3 Scripts/ci/tests/test_benchmark_contracts\.py "\$BENCH_BIN"\z}, 'actual CLI contract cases (not discovery)')
  require_command(bench, %r{\Apython3 Scripts/ci/check_benchmark_build_modes\.py --scratch-root "\$RUNNER_TEMP/benchmark-modes" --output benchmark-contract-results/build-modes --jobs [1-6]\z}, 'four distinct build modes and observable routes')
  aggregate = jobs.values.find { |job| job['name'] == 'CI Required' }
  expected_command = 'python3 Scripts/ci/check_required.py ' + REQUIRED_JOBS.join(' ')
  assert_policy(aggregate.fetch('steps').any? { |step| step['run'] == expected_command && step.dig('env', 'NEEDS_JSON') == '${{ toJSON(needs) }}' && !step.key?('if') && !step.key?('continue-on-error') }, 'aggregate must execute exact seven-job result check')
  validate_sanitizer_job(jobs.fetch('sanitizers'))
  standalone = workflows.fetch('sanitizers.yml')
  validate_sanitizer_job(standalone.fetch('jobs').fetch('sanitize'))
  bodies = JSON.parse(File.read(File.join(root, 'Scripts/ci/sanitizer_bodies.json'))).fetch('sanitizers')
  expected = Hash.new { |hash, key| hash[key] = [] }
  File.readlines(File.join(root, 'Scripts/ci/sanitizer_expected_bodies.txt')).each do |line|
    next if line.strip.empty? || line.start_with?('#')

    mode, id = line.split
    expected[mode] << id
  end
  assert_policy(bodies.keys.sort == %w[address thread], 'sanitizer manifest modes')
  bodies.each do |mode, entries|
    ids = entries.map { |entry| entry['id'] }
    assert_policy(!ids.empty? && ids.uniq == ids && ids.sort == expected[mode].sort, 'missing required actual instrumented body')
    entries.each do |entry|
      suite, method = entry['id'].split('/')
      assert_policy(entry['filter'] == '^ComprehensiveTests\\.' + suite + '/' + method + '\\(\\)', 'missing or inexact sanitizer filter')
      assert_policy(File.file?(File.join(root, entry['source'])), 'missing sanitizer source')
    end
  end
  driver = File.join(root, 'Scripts/ci/check_benchmark_build_modes.py')
  assert_policy(Digest::SHA256.file(driver).hexdigest == 'f300477f37f05cd9dedce983231a4debfede5311fb8ba0fea2b4e1f7c4963752', 'accepted V02 four-build/observable-route driver changed; root review required')
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
  validate_executable_contracts(ci, workflows, root)
rescue Errno::ENOENT, KeyError, JSON::ParserError, ArgumentError => error
  errors << ".github: #{error.message}"
end
unless errors.empty?
  warn errors.join("\n")
  exit 1
end
puts "Validated #{files.length} GitHub YAML/template files and workflow policies."
