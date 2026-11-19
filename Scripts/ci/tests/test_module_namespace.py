"""Guard module qualification against public declarations with the module's name."""

from pathlib import Path
import re
import sys
import tempfile
import unittest

CI = Path(__file__).resolve().parents[1]
ROOT = CI.parents[1]
sys.path.insert(0, str(CI))
from swift_lexing import _mask

DECLARATION_OR_BRACE = re.compile(
    r"[{}]|(?<![`\w.])(?:public|open)\s+"
    r"(?:(?:final|indirect|nonisolated(?:\s*\(\s*unsafe\s*\))?"
    r"|(?:private|fileprivate|internal|package)\s*\(\s*set\s*\))\s+)*"
    r"(?P<kind>actor|class|enum|struct|protocol|typealias|func|let|var|macro)\b\s*"
)
IDENTIFIER = re.compile(r"(?:`\w+`|\w+)")
BINDING = re.compile(rf"{IDENTIFIER.pattern}|\(\s*{IDENTIFIER.pattern}(?:\s*,\s*{IDENTIFIER.pattern})+\s*,?\s*\)")
FOLLOWING_BINDING = re.compile(rf",\s*(?P<binding>{BINDING.pattern})(?=\s*[:=,])")
BINDING_DELIMITER = re.compile(r"[(){}\[\],;\n]")


def binding_names(text, start):
    first = BINDING.match(text, start)
    if not first:
        return
    yield from IDENTIFIER.findall(first[0])

    # A top-level comma can continue the declaration. Ignore commas inside
    # calls, tuple values and closures before looking for the next binding.
    depth, consumed = 0, first.end()
    for delimiter in BINDING_DELIMITER.finditer(text, consumed):
        if delimiter.start() < consumed:
            continue
        punctuation = delimiter[0]
        if punctuation in "([{":
            depth += 1
        elif punctuation in ")]}" and depth:
            depth -= 1
        elif depth == 0:
            if punctuation in "\n;)}]":
                return
            next_binding = FOLLOWING_BINDING.match(text, delimiter.start())
            if next_binding:
                yield from IDENTIFIER.findall(next_binding["binding"])
                consumed = next_binding.end()


def module_name_declarations(source, module_name):
    """(line, kind) for explicit public/open file-scope names matching the module.

    Uses the comment and literal boundaries of swift_lexing._mask. Variable
    bindings cover single names, flat tuples and comma lists whose next
    separator follows the current initializer on the same logical line.
    """
    masked = _mask(source)
    depth = 0
    for token in DECLARATION_OR_BRACE.finditer(masked):
        if token[0] == "{":
            depth += 1
        elif token[0] == "}":
            depth -= 1
        elif depth == 0:
            kind = token["kind"]
            if kind in {"let", "var"}:
                names = binding_names(masked, token.end())
            else:
                name = IDENTIFIER.match(masked, token.end())
                names = [name[0]] if name else []
            if any(name.strip("`") == module_name for name in names):
                yield source.count("\n", 0, token.start()) + 1, kind


def module_name_violations(sources):
    violations = []
    for directory in sorted(path for path in sources.iterdir() if path.is_dir()):
        for path in sorted(directory.rglob("*.swift")):
            for line, kind in module_name_declarations(path.read_text(encoding="utf-8"), directory.name):
                violations.append(f"{path.relative_to(sources)}:{line}: {kind} {directory.name}")
    return violations


class ModuleNamespaceTests(unittest.TestCase):
    def test_public_declarations_with_module_name_are_detected(self):
        declarations = {
            "actor": "public actor Demo {}",
            "class": "open class Demo {}",
            "enum": "public indirect enum Demo { case next(Demo) }",
            "struct": "public struct Demo {}",
            "protocol": "public protocol Demo {}",
            "typealias": "public typealias `Demo` = Int",
            "func": "public func Demo() {}",
            "let": "public let Demo: Int = 0",
            "var": "public nonisolated(unsafe) var Demo = 0",
            "macro": "public macro Demo() = #externalMacro(module: \"Macros\", type: \"Demo\")",
        }
        for kind, source in declarations.items():
            with self.subTest(kind=kind):
                self.assertEqual(list(module_name_declarations(source, "Demo")), [(1, kind)])

    def test_attributes_indentation_and_conditional_compilation(self):
        source = """\
        #if canImport(Demo)
            @available(*, deprecated, message: "moved { here")
            public
                final class
                `Demo` {}
        #else
            final public class Demo {}
        #endif
        """
        self.assertEqual(list(module_name_declarations(source, "Demo")), [(3, "class"), (7, "class")])

    def test_private_nested_qualified_and_literal_controls(self):
        source = '''\
private struct Demo {}
package struct Demo {}
struct Demo {}
public struct DemoOther {}
public struct Container { public struct Demo {} }
public extension Demo.SomeType { struct Demo {} }
public extension Demo {}
public func helper() { struct Demo {} }
import struct Demo.Demo
// public typealias Demo = Other
/* public enum Demo {} */
let example = #"public actor Demo {}"#
let multiline = """
    public func Demo() {}
    """
'''
        self.assertEqual(list(module_name_declarations(source, "Demo")), [])

    def test_public_tuple_and_multi_bindings(self):
        bindings = (
            "public let (x, Demo) = (0, 1)",
            "public let first = 0, Demo = 1",
            "public let first = call(0, 1),\n    (Demo, x) = (2, 3)",
            "public let first: Result<Int, Error> = call(), Demo: Int = 1",
            "public private(set) var Demo = 1",
        )
        for source in bindings:
            with self.subTest(source=source):
                kind = "var" if "var " in source else "let"
                self.assertEqual(list(module_name_declarations(source, "Demo")), [(1, kind)])

    def test_mentions_in_other_bindings_do_not_declare_the_module_name(self):
        source = """\
        public let tuple = (other: 0, Demo: 1)
        public let generic: Pair<Int, Demo> = value
        public let closure = { let Demo = 0; return Demo }
        public let other = 0,
            module = Demo.self
        let keyword = `public`
        enum Demo {}
        """
        self.assertEqual(list(module_name_declarations(source, "Demo")), [])

    def test_scan_matches_each_module_own_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            sources = Path(temporary)
            for module in ("Demo", "Tools"):
                directory = sources / module / "Subdirectory"
                directory.mkdir(parents=True)
                (directory / "Entries.swift").write_text(
                    "public enum Demo {}\npublic enum Tools {}\n", encoding="utf-8"
                )
            self.assertEqual(
                module_name_violations(sources),
                ["Demo/Subdirectory/Entries.swift:1: enum Demo", "Tools/Subdirectory/Entries.swift:2: enum Tools"],
            )

    def test_sources_keep_module_qualification_available(self):
        self.assertEqual(module_name_violations(ROOT / "Sources"), [])
