"""Conservative syntax/API validator for candidate evaluation functions.

This is a defense-in-depth source restriction, not an OS security sandbox.
Run generated source only in a separately isolated worker before production.
"""
import ast
import builtins
import math
import time

ALLOWED_NODES = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
                 ast.Assign, ast.AugAssign, ast.If,
                 ast.Expr, ast.Name, ast.Load, ast.Store, ast.Constant,
                 ast.Tuple, ast.List, ast.BinOp, ast.UnaryOp, ast.BoolOp,
                 ast.Compare, ast.IfExp, ast.Subscript, ast.Slice, ast.Call,
                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                 ast.USub, ast.UAdd, ast.And, ast.Or, ast.Not,
                 ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                 ast.In, ast.NotIn, ast.Dict, ast.keyword, ast.Attribute, ast.Pass)
ALLOWED_CALLS = {"float", "int", "min", "max", "sum", "abs", "len"}
ALLOWED_VIEW_FIELDS = {"board", "captured", "current_player", "num_houses_per_player",
                      "total_seeds", "legal_actions", "terminal", "returns"}
TUPLE_VIEW_FIELDS = {"board", "captured", "legal_actions", "returns"}


def validate_source(source):
  """Validate the intentionally small, bounded candidate language.

  No imports, loops, comprehensions, recursion, dynamic attribute lookup, or
  filesystem/network-capable names are allowed. Evaluated source is still run
  under the pinned EoH hard-timeout subprocess.
  """
  if len(source) > 10000:
    return False, "source_too_large"
  try:
    tree = ast.parse(source)
  except SyntaxError:
    return False, "syntax_error"
  if any(not isinstance(node, ALLOWED_NODES + (ast.Slice,)) for node in ast.walk(tree)):
    return False, "disallowed_syntax"
  if sum(1 for _ in ast.walk(tree)) > 500:
    return False, "ast_too_large"
  if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
    return False, "expected_one_function"
  fn = tree.body[0]
  if fn.name != "evaluate_state" or [a.arg for a in fn.args.args] != ["state_view", "player_id"]:
    return False, "signature_mismatch"
  if fn.args.vararg or fn.args.kwarg or fn.args.defaults or fn.decorator_list:
    return False, "unsupported_function_features"
  local_types = {}

  def infer(node):
    """Tiny type checker: candidate arithmetic stays scalar; tuples are read-only inputs."""
    if isinstance(node, ast.Constant):
      return "scalar" if isinstance(node.value, (int, float, bool)) else "invalid"
    if isinstance(node, ast.Name):
      if node.id == "player_id": return "scalar"
      if node.id in local_types: return local_types[node.id]
      return "scalar" if node.id in ALLOWED_CALLS else "invalid"
    if isinstance(node, ast.Attribute):
      if not isinstance(node.value, ast.Name) or node.value.id != "state_view" or node.attr not in ALLOWED_VIEW_FIELDS:
        return "invalid"
      return "tuple" if node.attr in TUPLE_VIEW_FIELDS else "scalar"
    if isinstance(node, ast.Subscript):
      if infer(node.value) != "tuple": return "invalid"
      return "tuple" if isinstance(node.slice, ast.Slice) else "scalar" if infer(node.slice) == "scalar" else "invalid"
    if isinstance(node, ast.Call):
      name = node.func.id
      if name in {"sum", "min", "max", "len"}:
        if not node.args or any(infer(arg) not in {"scalar", "tuple"} for arg in node.args):
          return "invalid"
        return "scalar"
      if name in {"int", "float", "abs"}:
        return "scalar" if len(node.args) == 1 and infer(node.args[0]) in {"scalar", "bool"} else "invalid"
      return "invalid"
    if isinstance(node, ast.BinOp):
      return "scalar" if infer(node.left) in {"scalar", "bool"} and infer(node.right) in {"scalar", "bool"} else "invalid"
    if isinstance(node, ast.UnaryOp):
      return "scalar" if infer(node.operand) == "scalar" else "invalid"
    if isinstance(node, ast.IfExp):
      body_type, else_type = infer(node.body), infer(node.orelse)
      return body_type if infer(node.test) == "bool" and body_type == else_type and body_type in {"scalar", "tuple"} else "invalid"
    if isinstance(node, ast.Compare):
      return "bool" if infer(node.left) == "scalar" and all(infer(item) == "scalar" for item in node.comparators) else "invalid"
    if isinstance(node, ast.BoolOp):
      return "bool" if all(infer(value) == "bool" for value in node.values) else "invalid"
    return "invalid"

  def check_body(statements):
    for statement in statements:
      if isinstance(statement, ast.Return):
        if infer(statement.value) != "scalar": return False
      elif isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name): return False
        value_type = infer(statement.value)
        if value_type not in {"scalar", "tuple"}: return False
        local_types[statement.targets[0].id] = value_type
      elif isinstance(statement, ast.If):
        if infer(statement.test) != "bool" or not check_body(statement.body) or not check_body(statement.orelse): return False
      else:
        return False
    return True

  if not check_body(fn.body):
    return False, "unsafe_or_nonnumeric_expression"

  for node in ast.walk(tree):
    if not isinstance(node, ALLOWED_NODES + (ast.Slice,)):
      return False, "disallowed_syntax"
    if isinstance(node, ast.Attribute):
      if not isinstance(node.value, ast.Name) or node.value.id != "state_view" or node.attr not in ALLOWED_VIEW_FIELDS:
        return False, "disallowed_attribute"
      if isinstance(node.ctx, ast.Store):
        return False, "input_mutation"
    if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_CALLS):
      return False, "disallowed_call"
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
      if node.id not in {"evaluate_state", "state_view", "player_id"} | ALLOWED_CALLS:
        # Local variables are allowed when assigned somewhere in this function.
        assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if node.id not in assigned:
          return False, "unknown_name"
    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool, type(None))):
      return False, "unsupported_constant"
  return True, "passed"


def validate(source, views, *, seconds_per_call=0.05):
  started = time.perf_counter()
  try:
    ok, reason = validate_source(source)
    if not ok:
      return {"ok": False, "reason": reason}
    tree = ast.parse(source)
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in ALLOWED_CALLS}}
    exec(compile(tree, "<candidate>", "exec"), namespace, namespace)
    evaluator = namespace["evaluate_state"]
    results = []
    for view, player in views:
      before = repr(view)
      t0 = time.perf_counter()
      value = evaluator(view, player)
      elapsed = time.perf_counter() - t0
      if elapsed > seconds_per_call:
        return {"ok": False, "reason": "time_limit"}
      if repr(view) != before:
        return {"ok": False, "reason": "input_mutation"}
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {"ok": False, "reason": "non_numeric"}
      value = float(value)
      if not math.isfinite(value) or abs(value) > 10:
        return {"ok": False, "reason": "nonfinite_or_out_of_range"}
      results.append(value)
      if time.perf_counter() - started > seconds_per_call * max(1, len(views)):
        return {"ok": False, "reason": "total_time_limit"}
    if len(results) != len(views):
      return {"ok": False, "reason": "incomplete_fixtures"}
    return {"ok": True, "reason": "passed", "scores": results}
  except Exception as exc:
    return {"ok": False, "reason": "exception", "detail": type(exc).__name__}
