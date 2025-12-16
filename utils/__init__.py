import ast
# from lightning.pytorch.utilities import rank_zero_only


# @rank_zero_only
# def print_info(*values: object):
#     print(*values)


def safe_eval(input_string):
    try:
        # 尝试用 literal_eval 来解析字面量（如字符串、数字、列表、字典等）
        result = ast.literal_eval(input_string)
    except (ValueError, SyntaxError):
        # 如果解析失败，说明它是纯字符串或复杂表达式，直接返回
        result = input_string
    return result
