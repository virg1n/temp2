from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, List

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acl_pipeline.prompts import build_judge_batch_messages


DATA_DIR = ROOT / "judge_lora_dataset"
TRAIN_PATH = DATA_DIR / "train.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CONFIG_PATH = ROOT / "configs" / "default.yaml"

TRAIN_BACKUP_PATH = DATA_DIR / "train_before_gpt_synthetic_aug.jsonl"
AUGMENT_PATH = DATA_DIR / "train_synthetic_gpt_100_batches.jsonl"

BATCH_COUNT_TO_ADD = 100
BATCH_SIZE = 4

RUNTIME_KEYS = (
    "no_solution_reveal",
    "bug_localization",
    "usefulness",
    "socratic_style",
    "technical_accuracy",
    "task_quality",
    "task_is_valid_for_socratic",
    "hint_is_valid_for_socratic",
    "red_rejection_reason",
    "hint_rejection_reason",
)


@dataclass
class TaskSpec:
    family: str
    topic: str
    statement: str
    code: str
    observed_failure: str
    execution_status: str
    focus: str
    sample_call: str
    secondary: str = ""
    buggy_fragment: str = ""
    correct_fragment: str = ""
    error_name: str = ""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reward_weights() -> Dict[str, float]:
    return dict(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["judge"]["reward_weights"])


def ensure_backup() -> List[Dict[str, Any]]:
    if not TRAIN_BACKUP_PATH.exists():
        shutil.copyfile(TRAIN_PATH, TRAIN_BACKUP_PATH)
    return load_jsonl(TRAIN_BACKUP_PATH)


def traceback_assert(assert_line: str) -> str:
    return dedent(
        f"""\
        Traceback (most recent call last):
          File "/tmp/acl_task_generated/task.py", line 14, in <module>
            {assert_line}
        AssertionError
        """
    ).strip()


def traceback_nameerror(line_text: str, name: str) -> str:
    return dedent(
        f"""\
        Traceback (most recent call last):
          File "/tmp/acl_task_generated/task.py", line 11, in <module>
            {line_text}
        NameError: name '{name}' is not defined
        """
    ).strip()


def traceback_syntax(line_text: str, message: str) -> str:
    caret = "^".rjust(max(1, min(len(line_text), 24)))
    return dedent(
        f"""\
          File "/tmp/acl_task_generated/task.py", line 4
            {line_text}
            {caret}
        SyntaxError: {message}
        """
    ).strip()


def traceback_indentation(line_text: str, message: str) -> str:
    caret = "^".rjust(max(1, min(len(line_text), 24)))
    return dedent(
        f"""\
          File "/tmp/acl_task_generated/task.py", line 3
            {line_text}
            {caret}
        IndentationError: {message}
        """
    ).strip()


def window_sums_failed(idx: int) -> TaskSpec:
    fn = f"window_sums_{idx}"
    code = dedent(
        f"""\
        def {fn}(nums, width):
            if width <= 0 or width > len(nums):
                return []
            sums = []
            for start in range(len(nums) - width):
                sums.append(sum(nums[start:start + width]))
            return sums

        assert {fn}([1, 2, 3, 4], 2) == [3, 5, 7]
        assert {fn}([5, 5, 5], 3) == [15]
        """
    ).strip()
    return TaskSpec(
        family="window_sums_failed",
        topic="Loops and Iteration",
        statement=f"Write `{fn}` so it returns the sum of each consecutive window of length `width` from a list of numbers.",
        code=code,
        observed_failure=traceback_assert(f"assert {fn}([1, 2, 3, 4], 2) == [3, 5, 7]"),
        execution_status="failed",
        focus=fn,
        sample_call=f"{fn}([1, 2, 3, 4], 2)",
        secondary="start",
        buggy_fragment="range(len(nums) - width)",
        correct_fragment="range(len(nums) - width + 1)",
    )


def merge_priority_failed(idx: int) -> TaskSpec:
    fn = f"merge_priority_{idx}"
    code = dedent(
        f"""\
        def {fn}(left, right):
            merged = {{name: dict(record) for name, record in left.items()}}
            for name, record in right.items():
                if name not in merged:
                    merged[name] = dict(record)
                else:
                    merged[name].update(record)
                    if "status" in record:
                        merged[name]["status"] = record["status"]
            return merged

        left = {{"alice": {{"status": "active", "score": 3}}}}
        right = {{"alice": {{"status": "inactive", "score": 9}}, "bob": {{"status": "active", "score": 1}}}}
        expected = {{"alice": {{"status": "active", "score": 9}}, "bob": {{"status": "active", "score": 1}}}}
        assert {fn}(left, right) == expected
        """
    ).strip()
    return TaskSpec(
        family="merge_priority_failed",
        topic="Dictionaries and Sets",
        statement=f"Implement `{fn}` to merge two user-record dictionaries while preferring the status `'active'` when the same user appears in both mappings.",
        code=code,
        observed_failure=traceback_assert(f"assert {fn}(left, right) == expected"),
        execution_status="failed",
        focus=fn,
        sample_call=f"{fn}(left, right)",
        secondary="status",
        buggy_fragment='merged[name]["status"] = record["status"]',
        correct_fragment='preserve `"active"` when either side is active',
    )


def tag_index_failed(idx: int) -> TaskSpec:
    fn = f"build_tag_index_{idx}"
    code = dedent(
        f"""\
        def {fn}(groups):
            index = dict.fromkeys(groups.keys(), [])
            for label, tags in groups.items():
                for tag in tags:
                    index[label].append(tag.lower())
            return index

        groups = {{"fruit": ["Apple"], "tools": ["Saw"]}}
        assert {fn}(groups) == {{"fruit": ["apple"], "tools": ["saw"]}}
        """
    ).strip()
    return TaskSpec(
        family="tag_index_failed",
        topic="Mutable State and Aliasing",
        statement=f"Implement `{fn}` so each group name maps to its own list of normalized tags.",
        code=code,
        observed_failure=traceback_assert(f"assert {fn}(groups) == {{'fruit': ['apple'], 'tools': ['saw']}}"),
        execution_status="failed",
        focus=fn,
        sample_call=f"{fn}(groups)",
        secondary="index",
        buggy_fragment="dict.fromkeys(groups.keys(), [])",
        correct_fragment="{label: [] for label in groups}",
    )


def sort_books_failed(idx: int) -> TaskSpec:
    fn = f"sort_books_{idx}"
    code = dedent(
        f"""\
        def {fn}(books):
            return sorted(books, key=lambda book: (book["year"], book["title"].lower()))

        books = [
            {{"title": "Gamma", "year": 2022}},
            {{"title": "Alpha", "year": 2024}},
            {{"title": "Beta", "year": 2024}},
        ]
        assert [book["title"] for book in {fn}(books)] == ["Alpha", "Beta", "Gamma"]
        """
    ).strip()
    return TaskSpec(
        family="sort_books_failed",
        topic="Sorting and Key Functions",
        statement=f"Implement `{fn}` to sort books by descending publication year and then by case-insensitive title.",
        code=code,
        observed_failure=traceback_assert(f'assert [book["title"] for book in {fn}(books)] == ["Alpha", "Beta", "Gamma"]'),
        execution_status="failed",
        focus=fn,
        sample_call=f"{fn}(books)",
        secondary="year",
        buggy_fragment='key=lambda book: (book["year"], book["title"].lower())',
        correct_fragment='key=lambda book: (-book["year"], book["title"].lower())',
    )


def parse_point_failed(idx: int) -> TaskSpec:
    fn = f"parse_point_{idx}"
    code = dedent(
        f"""\
        def {fn}(text):
            x_str, y_str = text.split(",")
            return int(y_str), int(x_str)

        assert {fn}("3,5") == (3, 5)
        assert {fn}("10,20") == (10, 20)
        """
    ).strip()
    return TaskSpec(
        family="parse_point_failed",
        topic="Strings and Parsing",
        statement=f"Implement `{fn}` so it parses a comma-separated coordinate string into an `(x, y)` integer tuple.",
        code=code,
        observed_failure=traceback_assert(f'assert {fn}("3,5") == (3, 5)'),
        execution_status="failed",
        focus=fn,
        sample_call=f'{fn}("3,5")',
        secondary="x_str",
        buggy_fragment="return int(y_str), int(x_str)",
        correct_fragment="return int(x_str), int(y_str)",
    )


def make_scalers_failed(idx: int) -> TaskSpec:
    fn = f"make_scalers_{idx}"
    code = dedent(
        f"""\
        def {fn}():
            funcs = []
            for factor in [2, 3, 4]:
                funcs.append(lambda value: value * factor)
            return funcs

        double, triple, quadruple = {fn}()
        assert double(5) == 10
        assert triple(5) == 15
        assert quadruple(5) == 20
        """
    ).strip()
    return TaskSpec(
        family="make_scalers_failed",
        topic="Decorators and Closures",
        statement=f"Implement `{fn}` so it returns three independent scaling functions for factors 2, 3, and 4.",
        code=code,
        observed_failure=traceback_assert("assert double(5) == 10"),
        execution_status="failed",
        focus=fn,
        sample_call="double(5)",
        secondary="factor",
        buggy_fragment="lambda value: value * factor",
        correct_fragment="lambda value, factor=factor: value * factor",
    )


def wallet_transfer_failed(idx: int) -> TaskSpec:
    cls = f"Wallet_{idx}"
    code = dedent(
        f"""\
        class {cls}:
            def __init__(self, balance):
                self.balance = balance

            def transfer(self, amount, target):
                if amount > self.balance:
                    return False
                self.balance -= amount
                if target is self:
                    self.balance -= amount
                else:
                    target.balance += amount
                return True

        wallet = {cls}(100)
        assert wallet.transfer(20, wallet) is True
        assert wallet.balance == 100
        """
    ).strip()
    return TaskSpec(
        family="wallet_transfer_failed",
        topic="Objects and Classes",
        statement=f"Implement `{cls}.transfer` so self-transfers leave the balance unchanged while normal transfers still work.",
        code=code,
        observed_failure=traceback_assert("assert wallet.balance == 100"),
        execution_status="failed",
        focus=f"{cls}.transfer",
        sample_call="wallet.transfer(20, wallet)",
        secondary="balance",
        buggy_fragment="if target is self:\n            self.balance -= amount",
        correct_fragment="handle the self-transfer case without subtracting twice",
    )


def loyalty_discount_failed(idx: int) -> TaskSpec:
    fn = f"loyalty_discount_{idx}"
    code = dedent(
        f"""\
        def {fn}(tier, spend):
            if tier == "gold":
                if spend >= 500:
                    return 0.20
                elif spend >= 200:
                    return 0.10
                return 0.0
            if tier == "silver":
                if spend >= 200:
                    return 0.10
                elif spend >= 500:
                    return 0.15
                return 0.0
            return 0.0

        assert {fn}("gold", 600) == 0.20
        assert {fn}("silver", 700) == 0.15
        """
    ).strip()
    return TaskSpec(
        family="loyalty_discount_failed",
        topic="Conditional Statements",
        statement=f"Implement `{fn}` so discount thresholds are applied correctly for each loyalty tier.",
        code=code,
        observed_failure=traceback_assert(f'assert {fn}("silver", 700) == 0.15'),
        execution_status="failed",
        focus=fn,
        sample_call=f'{fn}("silver", 700)',
        secondary="spend",
        buggy_fragment='if spend >= 200:\n            return 0.10\n        elif spend >= 500:',
        correct_fragment="check the 500 threshold before the 200 threshold",
    )


def cyclic_iterator_failed(idx: int) -> TaskSpec:
    cls = f"CyclicIterator_{idx}"
    code = dedent(
        f"""\
        class {cls}:
            def __init__(self, values):
                self.values = list(values)
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                value = self.values[self.index]
                self.index = (self.index + 1) % len(self.values)
                return self.values[self.index]

        iterator = {cls}([1, 2, 3])
        assert next(iterator) == 1
        assert next(iterator) == 2
        """
    ).strip()
    return TaskSpec(
        family="cyclic_iterator_failed",
        topic="Inheritance, MRO, and Dunder Methods",
        statement=f"Implement `{cls}` so repeated calls to `next()` cycle through the stored values in order.",
        code=code,
        observed_failure=traceback_assert("assert next(iterator) == 1"),
        execution_status="failed",
        focus=f"{cls}.__next__",
        sample_call="next(iterator)",
        secondary="index",
        buggy_fragment="return self.values[self.index]",
        correct_fragment="return value",
    )


def count_even_squares_failed(idx: int) -> TaskSpec:
    fn = f"count_even_squares_{idx}"
    code = dedent(
        f"""\
        def {fn}(data):
            total = 0
            for item in data:
                if isinstance(item, list):
                    total += {fn}(item)
                elif isinstance(item, int) and item % 2 == 0:
                    total += 1
            return total

        assert {fn}([4, [8, 9], 16]) == 2
        assert {fn}([1, [2, [25]], 36]) == 1
        """
    ).strip()
    return TaskSpec(
        family="count_even_squares_failed",
        topic="Recursion and Generators",
        statement=f"Implement `{fn}` so it recursively counts only even integers that are also perfect squares.",
        code=code,
        observed_failure=traceback_assert(f"assert {fn}([4, [8, 9], 16]) == 2"),
        execution_status="failed",
        focus=fn,
        sample_call=f"{fn}([4, [8, 9], 16])",
        secondary="item",
        buggy_fragment="elif isinstance(item, int) and item % 2 == 0:\n            total += 1",
        correct_fragment="check both evenness and perfect-square status before incrementing",
    )


def parse_positive_ints_failed(idx: int) -> TaskSpec:
    cls = f"TokenError_{idx}"
    fn = f"parse_positive_ints_{idx}"
    code = dedent(
        f"""\
        class {cls}(Exception):
            pass

        def {fn}(tokens):
            values = []
            for token in tokens:
                if not token.isdigit():
                    return values
                values.append(int(token))
            return values

        try:
            {fn}(["3", "bad"])
            assert False, "Expected TokenError"
        except {cls}:
            pass
        """
    ).strip()
    return TaskSpec(
        family="parse_positive_ints_failed",
        topic="Exception Handling and Custom Exceptions",
        statement=f"Implement `{fn}` so it raises `{cls}` when any token is not a positive integer string.",
        code=code,
        observed_failure=traceback_assert('assert False, "Expected TokenError"'),
        execution_status="failed",
        focus=fn,
        sample_call=f'{fn}(["3", "bad"])',
        secondary=cls,
        buggy_fragment="return values",
        correct_fragment=f"raise {cls}(...)",
    )


def triangle_passed(idx: int) -> TaskSpec:
    fn = f"classify_triangle_ok_{idx}"
    code = dedent(
        f"""\
        def {fn}(a, b, c):
            a, b, c = sorted([a, b, c])
            if a <= 0 or a + b <= c:
                return "invalid"
            if a == b == c:
                return "equilateral"
            if a * a + b * b == c * c:
                return "right-angled"
            if a == b or b == c:
                return "isosceles"
            return "scalene"

        assert {fn}(3, 4, 5) == "right-angled"
        assert {fn}(2, 2, 3) == "isosceles"
        assert {fn}(5, 5, 11) == "invalid"
        """
    ).strip()
    return TaskSpec(
        family="triangle_passed",
        topic="Conditional Statements",
        statement=f"Implement `{fn}` so it classifies triangles as invalid, equilateral, isosceles, right-angled, or scalene.",
        code=code,
        observed_failure="Program exited successfully. No failing assertion or runtime error was reproduced.",
        execution_status="passed",
        focus=fn,
        sample_call=f"{fn}(3, 4, 5)",
    )


def merge_priority_passed(idx: int) -> TaskSpec:
    fn = f"merge_priority_ok_{idx}"
    code = dedent(
        f"""\
        def {fn}(left, right):
            merged = {{name: dict(record) for name, record in left.items()}}
            for name, record in right.items():
                if name not in merged:
                    merged[name] = dict(record)
                    continue
                merged[name].update(record)
                if merged[name].get("status") == "inactive" and (
                    left.get(name, {{}}).get("status") == "active" or record.get("status") == "active"
                ):
                    merged[name]["status"] = "active"
            return merged

        left = {{"alice": {{"status": "active", "score": 3}}}}
        right = {{"alice": {{"status": "inactive", "score": 9}}, "bob": {{"status": "active", "score": 1}}}}
        expected = {{"alice": {{"status": "active", "score": 9}}, "bob": {{"status": "active", "score": 1}}}}
        assert {fn}(left, right) == expected
        """
    ).strip()
    return TaskSpec(
        family="merge_priority_passed",
        topic="Dictionaries and Sets",
        statement=f"Implement `{fn}` so merge behavior preserves an `'active'` status when either input side is active.",
        code=code,
        observed_failure="Program exited successfully. No failing assertion or runtime error was reproduced.",
        execution_status="passed",
        focus=fn,
        sample_call=f"{fn}(left, right)",
    )


def transfer_passed(idx: int) -> TaskSpec:
    cls = f"WalletOk_{idx}"
    code = dedent(
        f"""\
        class {cls}:
            def __init__(self, balance):
                self.balance = balance

            def transfer(self, amount, target):
                if amount > self.balance:
                    return False
                if target is self:
                    return True
                self.balance -= amount
                target.balance += amount
                return True

        wallet = {cls}(100)
        other = {cls}(10)
        assert wallet.transfer(20, other) is True
        assert wallet.balance == 80
        assert other.balance == 30
        assert wallet.transfer(10, wallet) is True
        assert wallet.balance == 80
        """
    ).strip()
    return TaskSpec(
        family="transfer_passed",
        topic="Objects and Classes",
        statement=f"Implement `{cls}.transfer` so normal transfers move funds and self-transfers keep balances unchanged.",
        code=code,
        observed_failure="Program exited successfully. No failing assertion or runtime error was reproduced.",
        execution_status="passed",
        focus=f"{cls}.transfer",
        sample_call="wallet.transfer(10, wallet)",
    )


def window_sums_passed(idx: int) -> TaskSpec:
    fn = f"window_sums_ok_{idx}"
    code = dedent(
        f"""\
        def {fn}(nums, width):
            if width <= 0 or width > len(nums):
                return []
            sums = []
            for start in range(len(nums) - width + 1):
                sums.append(sum(nums[start:start + width]))
            return sums

        assert {fn}([1, 2, 3, 4], 2) == [3, 5, 7]
        assert {fn}([5, 5, 5], 3) == [15]
        """
    ).strip()
    return TaskSpec(
        family="window_sums_passed",
        topic="Loops and Iteration",
        statement=f"Implement `{fn}` so it returns the sum of each consecutive fixed-width window.",
        code=code,
        observed_failure="Program exited successfully. No failing assertion or runtime error was reproduced.",
        execution_status="passed",
        focus=fn,
        sample_call=f"{fn}([1, 2, 3, 4], 2)",
    )


def parse_point_passed(idx: int) -> TaskSpec:
    fn = f"parse_point_ok_{idx}"
    code = dedent(
        f"""\
        def {fn}(text):
            x_str, y_str = text.split(",")
            return int(x_str), int(y_str)

        assert {fn}("3,5") == (3, 5)
        assert {fn}("10,20") == (10, 20)
        """
    ).strip()
    return TaskSpec(
        family="parse_point_passed",
        topic="Strings and Parsing",
        statement=f"Implement `{fn}` so it parses a comma-separated coordinate string into an `(x, y)` integer tuple.",
        code=code,
        observed_failure="Program exited successfully. No failing assertion or runtime error was reproduced.",
        execution_status="passed",
        focus=fn,
        sample_call=f'{fn}("3,5")',
    )


def convert_speed_passed(idx: int) -> TaskSpec:
    fn = f"convert_speed_ok_{idx}"
    code = dedent(
        f"""\
        def {fn}(value, unit):
            factors = {{"m/s": 1.0, "km/h": 1000 / 3600, "mph": 1609.34 / 3600}}
            return round(value * factors[unit], 3)

        assert {fn}(3.6, "km/h") == 1.0
        assert {fn}(1, "m/s") == 1.0
        """
    ).strip()
    return TaskSpec(
        family="convert_speed_passed",
        topic="Arithmetic and Type Conversion",
        statement=f"Implement `{fn}` so it converts a numeric speed to meters per second for a supported unit label.",
        code=code,
        observed_failure="Program exited successfully. No failing assertion or runtime error was reproduced.",
        execution_status="passed",
        focus=fn,
        sample_call=f'{fn}(3.6, "km/h")',
    )


def running_average_nameerror(idx: int) -> TaskSpec:
    fn = f"running_average_{idx}"
    code = dedent(
        f"""\
        def {fn}(nums):
            running_total = 0
            for value in nums:
                running_total += value
            return total_sum / len(nums)

        assert round({fn}([2, 4, 6]), 2) == 4.0
        """
    ).strip()
    return TaskSpec(
        family="running_average_nameerror",
        topic="Arithmetic and Type Conversion",
        statement=f"Implement `{fn}` so it computes the arithmetic mean of a list of numbers.",
        code=code,
        observed_failure=traceback_nameerror(f"assert round({fn}([2, 4, 6]), 2) == 4.0", "total_sum"),
        execution_status="nameerror",
        focus=fn,
        sample_call=f"{fn}([2, 4, 6])",
        error_name="total_sum",
    )


def render_receipt_nameerror(idx: int) -> TaskSpec:
    fn = f"render_receipt_{idx}"
    code = dedent(
        f"""\
        def {fn}(items):
            lines = []
            for name, price in items:
                lines.append(f"{{name}}: ${{price}}")
            return "\\n".join(formatted_lines)

        assert "tea" in {fn}([("tea", 3), ("cake", 5)])
        """
    ).strip()
    return TaskSpec(
        family="render_receipt_nameerror",
        topic="Functions and Scope",
        statement=f"Implement `{fn}` so it formats `(name, price)` pairs into a receipt string.",
        code=code,
        observed_failure=traceback_nameerror('return "\\n".join(formatted_lines)', "formatted_lines"),
        execution_status="nameerror",
        focus=fn,
        sample_call=f'{fn}([("tea", 3), ("cake", 5)])',
        error_name="formatted_lines",
    )


def diagonal_total_nameerror(idx: int) -> TaskSpec:
    fn = f"diagonal_total_{idx}"
    code = dedent(
        f"""\
        def {fn}(rows):
            total = 0
            for index, row in enumerate(rows):
                total += row[index]
            return diag_total

        assert {fn}([[1, 2], [3, 4]]) == 5
        """
    ).strip()
    return TaskSpec(
        family="diagonal_total_nameerror",
        topic="NumPy and Linear Algebra",
        statement=f"Implement `{fn}` so it returns the sum of the main diagonal of a square matrix represented as nested lists.",
        code=code,
        observed_failure=traceback_nameerror("return diag_total", "diag_total"),
        execution_status="nameerror",
        focus=fn,
        sample_call=f"{fn}([[1, 2], [3, 4]])",
        error_name="diag_total",
    )


def histogram_nameerror(idx: int) -> TaskSpec:
    fn = f"build_histogram_{idx}"
    code = dedent(
        f"""\
        def {fn}(words):
            counts = {{}}
            for word in words:
                current = counts.get(word, 0)
                counts[word] = current + 1
            return totals

        assert {fn}(["a", "b", "a"]) == {{"a": 2, "b": 1}}
        """
    ).strip()
    return TaskSpec(
        family="histogram_nameerror",
        topic="Dictionaries and Sets",
        statement=f"Implement `{fn}` so it counts occurrences of each word in the input sequence.",
        code=code,
        observed_failure=traceback_nameerror("return totals", "totals"),
        execution_status="nameerror",
        focus=fn,
        sample_call=f'{fn}(["a", "b", "a"])',
        error_name="totals",
    )


def missing_colon_syntax(idx: int) -> TaskSpec:
    fn = f"validate_age_{idx}"
    bad_line = "if age < 0"
    code = dedent(
        f"""\
        def {fn}(age):
            {bad_line}
                raise ValueError("age must be non-negative")
            return age

        assert {fn}(3) == 3
        """
    ).strip()
    return TaskSpec(
        family="missing_colon_syntax",
        topic="Conditional Statements",
        statement=f"Implement `{fn}` so it rejects negative ages and otherwise returns the original value.",
        code=code,
        observed_failure=traceback_syntax(bad_line, "expected ':'"),
        execution_status="syntax_error",
        focus=fn,
        sample_call=f"{fn}(3)",
    )


def escaped_newline_syntax(idx: int) -> TaskSpec:
    fn = f"run_check_{idx}"
    bad_line = 'assert value == 3\\n'
    code = dedent(
        f"""\
        def {fn}():
            value = 3
            {bad_line}

        {fn}()
        """
    ).strip()
    return TaskSpec(
        family="escaped_newline_syntax",
        topic="Functions and Scope",
        statement=f"Implement `{fn}` so its internal assertion is written as valid Python.",
        code=code,
        observed_failure=traceback_syntax(bad_line, "unexpected character after line continuation character"),
        execution_status="syntax_error",
        focus=fn,
        sample_call=f"{fn}()",
    )


def unmatched_bracket_syntax(idx: int) -> TaskSpec:
    fn = f"pair_total_{idx}"
    bad_line = "pairs = [(1, 2), (3, 4]"
    code = dedent(
        f"""\
        def {fn}():
            {bad_line}
            return sum(a + b for a, b in pairs)

        assert {fn}() == 10
        """
    ).strip()
    return TaskSpec(
        family="unmatched_bracket_syntax",
        topic="Lists, Tuples, and Slicing",
        statement=f"Implement `{fn}` so it sums a list of integer pairs.",
        code=code,
        observed_failure=traceback_syntax(bad_line, "closing parenthesis ']' does not match opening parenthesis '('"),
        execution_status="syntax_error",
        focus=fn,
        sample_call=f"{fn}()",
    )


def missing_indent_error(idx: int) -> TaskSpec:
    fn = f"normalize_user_{idx}"
    bad_line = 'print(name.strip().lower())'
    code = dedent(
        f"""\
        def {fn}(name):
        {bad_line}
            return name

        assert {fn}(" Alice ") == " Alice "
        """
    ).strip()
    return TaskSpec(
        family="missing_indent_error",
        topic="Functions and Scope",
        statement=f"Implement `{fn}` so it normalizes a user name before returning it.",
        code=code,
        observed_failure=traceback_indentation(bad_line, "expected an indented block after function definition on line 1"),
        execution_status="indentation_error",
        focus=fn,
        sample_call=f'{fn}(" Alice ")',
    )


def misaligned_indent_error(idx: int) -> TaskSpec:
    fn = f"accumulate_{idx}"
    bad_line = "  for value in values:"
    code = dedent(
        f"""\
        def {fn}(values):
            total = 0
          for value in values:
                total += value
            return total

        assert {fn}([1, 2, 3]) == 6
        """
    ).strip()
    return TaskSpec(
        family="misaligned_indent_error",
        topic="Loops and Iteration",
        statement=f"Implement `{fn}` so it accumulates all values in the input sequence.",
        code=code,
        observed_failure=traceback_indentation(bad_line, "unindent does not match any outer indentation level"),
        execution_status="indentation_error",
        focus=fn,
        sample_call=f"{fn}([1, 2, 3])",
    )


FAILED_BUILDERS: List[Callable[[int], TaskSpec]] = [
    window_sums_failed,
    merge_priority_failed,
    tag_index_failed,
    sort_books_failed,
    parse_point_failed,
    make_scalers_failed,
    wallet_transfer_failed,
    loyalty_discount_failed,
    cyclic_iterator_failed,
    count_even_squares_failed,
    parse_positive_ints_failed,
]

PASSED_BUILDERS: List[Callable[[int], TaskSpec]] = [
    triangle_passed,
    merge_priority_passed,
    transfer_passed,
    window_sums_passed,
    parse_point_passed,
    convert_speed_passed,
]

NAMEERROR_BUILDERS: List[Callable[[int], TaskSpec]] = [
    running_average_nameerror,
    render_receipt_nameerror,
    diagonal_total_nameerror,
    histogram_nameerror,
]

SYNTAX_BUILDERS: List[Callable[[int], TaskSpec]] = [
    missing_colon_syntax,
    escaped_newline_syntax,
    unmatched_bracket_syntax,
]

INDENT_BUILDERS: List[Callable[[int], TaskSpec]] = [
    missing_indent_error,
    misaligned_indent_error,
]


def ordered_labels(**kwargs: Any) -> Dict[str, Any]:
    return {key: kwargs[key] for key in RUNTIME_KEYS}


def quality_bucket_for_mode(mode: str) -> str:
    mapping = {
        "excellent": "excellent",
        "strong": "strong",
        "mixed": "mixed",
        "hallucinated": "bad",
        "corrupted": "corrupted",
        "reveal": "reveal",
        "passed_aware": "passed-aware",
        "missed_passed": "bad",
        "nameerror_aware": "strong",
        "nameerror_missed": "bad",
        "syntax_aware": "mixed",
        "syntax_missed": "bad",
        "indentation_aware": "mixed",
        "indentation_missed": "bad",
    }
    return mapping[mode]


def labels_for_mode(mode: str, status: str) -> Dict[str, Any]:
    if mode == "excellent":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=9.0,
            usefulness=9.0,
            socratic_style=9.0,
            technical_accuracy=9.0,
            task_quality=8.0,
            task_is_valid_for_socratic=True,
            hint_is_valid_for_socratic=True,
            red_rejection_reason="",
            hint_rejection_reason="",
        )
    if mode == "strong":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=7.0,
            usefulness=7.0,
            socratic_style=8.0,
            technical_accuracy=8.0,
            task_quality=8.0,
            task_is_valid_for_socratic=True,
            hint_is_valid_for_socratic=True,
            red_rejection_reason="",
            hint_rejection_reason="",
        )
    if mode == "mixed":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=5.0,
            usefulness=5.0,
            socratic_style=7.0,
            technical_accuracy=6.0,
            task_quality=8.0,
            task_is_valid_for_socratic=True,
            hint_is_valid_for_socratic=True,
            red_rejection_reason="",
            hint_rejection_reason="",
        )
    if mode == "hallucinated":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=1.0,
            usefulness=1.0,
            socratic_style=4.0,
            technical_accuracy=1.0,
            task_quality=8.0,
            task_is_valid_for_socratic=True,
            hint_is_valid_for_socratic=False,
            red_rejection_reason="",
            hint_rejection_reason="hallucinated_or_ungrounded",
        )
    if mode == "corrupted":
        return ordered_labels(
            no_solution_reveal=0.0,
            bug_localization=0.0,
            usefulness=0.0,
            socratic_style=0.0,
            technical_accuracy=0.0,
            task_quality=8.0,
            task_is_valid_for_socratic=True,
            hint_is_valid_for_socratic=False,
            red_rejection_reason="",
            hint_rejection_reason="malformed_hint_output",
        )
    if mode == "reveal":
        return ordered_labels(
            no_solution_reveal=1.0,
            bug_localization=8.0,
            usefulness=7.0,
            socratic_style=1.0,
            technical_accuracy=8.0,
            task_quality=8.0,
            task_is_valid_for_socratic=True,
            hint_is_valid_for_socratic=False,
            red_rejection_reason="",
            hint_rejection_reason="solution_reveal",
        )
    if mode == "passed_aware":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=0.0,
            usefulness=7.0,
            socratic_style=7.0,
            technical_accuracy=10.0,
            task_quality=2.0,
            task_is_valid_for_socratic=False,
            hint_is_valid_for_socratic=True,
            red_rejection_reason="already_correct_code",
            hint_rejection_reason="",
        )
    if mode == "missed_passed":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=0.0,
            usefulness=1.0,
            socratic_style=4.0,
            technical_accuracy=0.0,
            task_quality=2.0,
            task_is_valid_for_socratic=False,
            hint_is_valid_for_socratic=False,
            red_rejection_reason="already_correct_code",
            hint_rejection_reason="missed_passed_execution",
        )
    if mode == "nameerror_aware":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=7.0,
            usefulness=7.0,
            socratic_style=7.0,
            technical_accuracy=9.0,
            task_quality=7.0,
            task_is_valid_for_socratic=True,
            hint_is_valid_for_socratic=True,
            red_rejection_reason="",
            hint_rejection_reason="",
        )
    if mode == "nameerror_missed":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=2.0,
            usefulness=2.0,
            socratic_style=5.0,
            technical_accuracy=2.0,
            task_quality=7.0,
            task_is_valid_for_socratic=True,
            hint_is_valid_for_socratic=False,
            red_rejection_reason="",
            hint_rejection_reason="not_grounded_in_nameerror",
        )
    if mode == "syntax_aware":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=6.0,
            usefulness=6.0,
            socratic_style=6.0,
            technical_accuracy=9.0,
            task_quality=2.0,
            task_is_valid_for_socratic=False,
            hint_is_valid_for_socratic=True,
            red_rejection_reason="unrelated_syntax_error",
            hint_rejection_reason="",
        )
    if mode == "syntax_missed":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=1.0,
            usefulness=1.0,
            socratic_style=4.0,
            technical_accuracy=1.0,
            task_quality=2.0,
            task_is_valid_for_socratic=False,
            hint_is_valid_for_socratic=False,
            red_rejection_reason="unrelated_syntax_error",
            hint_rejection_reason="missed_syntax_issue",
        )
    if mode == "indentation_aware":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=6.0,
            usefulness=6.0,
            socratic_style=6.0,
            technical_accuracy=9.0,
            task_quality=2.0,
            task_is_valid_for_socratic=False,
            hint_is_valid_for_socratic=True,
            red_rejection_reason="unrelated_indentation_error",
            hint_rejection_reason="",
        )
    if mode == "indentation_missed":
        return ordered_labels(
            no_solution_reveal=10.0,
            bug_localization=1.0,
            usefulness=1.0,
            socratic_style=4.0,
            technical_accuracy=1.0,
            task_quality=2.0,
            task_is_valid_for_socratic=False,
            hint_is_valid_for_socratic=False,
            red_rejection_reason="unrelated_indentation_error",
            hint_rejection_reason="missed_indentation_issue",
        )
    raise ValueError(f"Unsupported mode: {mode} for status {status}")


def hint_for_mode(task: TaskSpec, mode: str, idx: int) -> str:
    if mode == "excellent":
        return (
            f"What does `{task.sample_call}` produce right now, and what does the failing assertion expect? "
            f"In `{task.focus}`, which value or branch involving `{task.secondary or task.focus}` changes immediately before the failure?"
        )
    if mode == "strong":
        return (
            f"Trace `{task.focus}` with `{task.sample_call}`. "
            f"Which intermediate value diverges from the expected result before the assertion fails?"
        )
    if mode == "mixed":
        return (
            f"Walk through `{task.focus}` step by step. "
            f"Which value seems off before the current failure appears?"
        )
    if mode == "hallucinated":
        return (
            f"Check whether `cursor_cache_{idx}` is refreshed inside `sync_cursor_{idx}()` "
            f"before `emit_cursor_{idx}()` runs."
        )
    if mode == "corrupted":
        return f"Whаt dоes `{task.focus}` return here?? `broken_{idx}"
    if mode == "reveal":
        return (
            f"The fix is to replace `{task.buggy_fragment}` with `{task.correct_fragment}` in `{task.focus}`."
        )
    if mode == "passed_aware":
        return (
            f"The run says no failing assertion or runtime error was reproduced. "
            f"Before debugging `{task.focus}`, which test is actually failing, if any?"
        )
    if mode == "missed_passed":
        return (
            f"What do you expect `{task.sample_call}` to return? "
            f"Trace `{task.focus}` step by step and compare it with the intended behavior."
        )
    if mode == "nameerror_aware":
        return (
            f"Which name is undefined in the traceback: `{task.error_name}` or something else? "
            f"Where should that name be defined before `{task.focus}` reaches that line?"
        )
    if mode == "nameerror_missed":
        return (
            f"Trace `{task.focus}` with `{task.sample_call}` and see which branch seems wrong before a result is returned."
        )
    if mode == "syntax_aware":
        return (
            f"The traceback stops at parsing before any test runs. "
            f"Which line near `{task.focus}` is invalid Python syntax?"
        )
    if mode == "syntax_missed":
        return f"Trace `{task.focus}` with a small input and see where the logic diverges."
    if mode == "indentation_aware":
        return (
            f"The traceback shows an indentation problem before execution begins. "
            f"Which block near `{task.focus}` is indented incorrectly?"
        )
    if mode == "indentation_missed":
        return f"Walk through `{task.focus}` and compare each branch with the expected behavior."
    raise ValueError(f"Unsupported mode: {mode}")


def select_task_and_mode(item_index: int) -> tuple[TaskSpec, str]:
    slot = item_index % 10
    if slot == 0:
        return FAILED_BUILDERS[item_index % len(FAILED_BUILDERS)](item_index), "excellent"
    if slot == 1:
        return FAILED_BUILDERS[item_index % len(FAILED_BUILDERS)](item_index), "strong"
    if slot == 2:
        return FAILED_BUILDERS[item_index % len(FAILED_BUILDERS)](item_index), "mixed"
    if slot == 3:
        mode = "corrupted" if item_index % 20 == 3 else "hallucinated"
        return FAILED_BUILDERS[item_index % len(FAILED_BUILDERS)](item_index), mode
    if slot == 4:
        return FAILED_BUILDERS[item_index % len(FAILED_BUILDERS)](item_index), "reveal"
    if slot == 5:
        return PASSED_BUILDERS[item_index % len(PASSED_BUILDERS)](item_index), "passed_aware"
    if slot == 6:
        return PASSED_BUILDERS[item_index % len(PASSED_BUILDERS)](item_index), "missed_passed"
    if slot == 7:
        mode = "nameerror_aware" if item_index % 4 in {1, 2} else "nameerror_missed"
        return NAMEERROR_BUILDERS[item_index % len(NAMEERROR_BUILDERS)](item_index), mode
    if slot == 8:
        if item_index % 4 in {0, 1}:
            mode = "syntax_aware" if item_index % 8 == 0 else "syntax_missed"
            return SYNTAX_BUILDERS[item_index % len(SYNTAX_BUILDERS)](item_index), mode
        mode = "indentation_aware" if item_index % 8 == 2 else "indentation_missed"
        return INDENT_BUILDERS[item_index % len(INDENT_BUILDERS)](item_index), mode
    return FAILED_BUILDERS[item_index % len(FAILED_BUILDERS)](item_index), "excellent"


def build_item(item_index: int) -> Dict[str, Any]:
    task, mode = select_task_and_mode(item_index)
    hint = hint_for_mode(task, mode, item_index)[:1800]
    labels = labels_for_mode(mode, task.execution_status)
    note = f"synthetic_gpt:{task.execution_status}:{mode}:{task.family}"
    return {
        "sample_id": f"synthetic-gpt-batch-{(item_index // 4) + 1:03d}-item-{(item_index % 4) + 1}",
        "source_group": "synthetic_gpt",
        "source_detail": {
            "generator": "codex_local",
            "family": task.family,
            "mode": mode,
            "item_index": item_index,
        },
        "topic": task.topic,
        "quality_bucket": quality_bucket_for_mode(mode),
        "label_notes": [note],
        "code_hash": sha256_text(task.code),
        "hint_hash": sha256_text(hint),
        "original_execution_status": task.execution_status,
        "runtime_item": {
            "statement": task.statement,
            "code": task.code,
            "observed_failure": task.observed_failure,
            "execution_status": task.execution_status,
            "assistant_response": hint,
        },
        "runtime_labels": labels,
    }


def build_batches(weights: Dict[str, float]) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    for batch_index in range(BATCH_COUNT_TO_ADD):
        items = [build_item(batch_index * BATCH_SIZE + offset) for offset in range(BATCH_SIZE)]
        payload = [item["runtime_item"] for item in items]
        labels = [item["runtime_labels"] for item in items]
        messages = build_judge_batch_messages(payload, weights)
        messages.append({"role": "assistant", "content": json.dumps(labels, ensure_ascii=False)})
        batches.append(
            {
                "batch_id": f"train-synthetic-gpt-{batch_index + 1:04d}",
                "split": "train",
                "batch_size": BATCH_SIZE,
                "sample_ids": [item["sample_id"] for item in items],
                "topics": [item["topic"] for item in items],
                "source_groups": [item["source_group"] for item in items],
                "quality_buckets": [item["quality_bucket"] for item in items],
                "items": items,
                "messages": messages,
            }
        )
    return batches


def update_manifest(base_train_count: int, base_train_items: int, aug_batches: List[Dict[str, Any]]) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["base_train_batch_count"] = base_train_count
    manifest["base_train_item_count"] = base_train_items
    manifest["synthetic_gpt_train_batch_count_added"] = len(aug_batches)
    manifest["synthetic_gpt_train_item_count_added"] = len(aug_batches) * BATCH_SIZE
    manifest["augmented_train_batch_count"] = base_train_count + len(aug_batches)
    manifest["augmented_train_item_count"] = base_train_items + (len(aug_batches) * BATCH_SIZE)
    manifest["train_pre_augmentation_backup"] = str(TRAIN_BACKUP_PATH.relative_to(ROOT)).replace("\\", "/")
    manifest["train_augmentation_file"] = str(AUGMENT_PATH.relative_to(ROOT)).replace("\\", "/")
    manifest.setdefault("notes", []).append(
        "train.jsonl has been augmented with 100 synthetic GPT-authored N=4 batches; full/val files remain unchanged"
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    base_train = ensure_backup()
    weights = reward_weights()
    aug_batches = build_batches(weights)
    write_jsonl(AUGMENT_PATH, aug_batches)
    write_jsonl(TRAIN_PATH, base_train + aug_batches)
    base_train_items = sum(int(row.get("batch_size", 0)) for row in base_train)
    update_manifest(len(base_train), base_train_items, aug_batches)


if __name__ == "__main__":
    main()
