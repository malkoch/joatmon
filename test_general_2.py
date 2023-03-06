import random

add = lambda a, b: a + b
sub = lambda a, b: a - b
mul = lambda a, b: a * b
div = lambda a, b: a / b if a % b == 0 else 0 / 0

operations = [(add, '+'),
              (sub, '-'),
              (mul, '*'),
              (div, '/')]


def one_from_the_top():
    return [25, 50, 75][random.randint(0, 2)]


def one_of_the_others():
    return random.randint(1, 10)


def evaluate(stack):
    try:
        total = 0
        last_oper = add
        for item in stack:
            if type(item) is int:
                total = last_oper(total, item)
            else:
                last_oper = item[0]

        return total
    except:
        return 0


def repr_stack(stack):
    reps = [str(item) if type(item) is int else item[1] for item in stack]
    return ' '.join(reps)


def solve(target, numbers):

    def recurse(stack, nums):
        for n in range(len(nums)):
            stack.append(nums[n])

            remaining = nums[:n] + nums[n + 1:]

            if evaluate(stack) == target:
                print(repr_stack(stack))

            if len(remaining) > 0:
                for op in operations:
                    stack.append(op)
                    stack = recurse(stack, remaining)
                    stack = stack[:-1]

            stack = stack[:-1]

        return stack

    recurse([], numbers)


target = random.randint(100, 1000)

numbers = [one_from_the_top()] + [one_of_the_others() for i in range(5)]

print("Target: {0} using {1}".format(target, numbers))

solve(target, numbers)
