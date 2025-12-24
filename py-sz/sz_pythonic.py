#1. F-Strings for Clean Formatting
# Forget .format() or % interpolation. F-strings are faster and much easier to read.
# You can even perform math or call methods directly inside the braces.
def demo_001():
    name, age = "Alice", 30
    print(f"{name.upper()} will be {age + 1} next year.")

# 2. Unpack Sequences Directly
# Instead of:
def demo_002():
    point = (10, 50)
    x = point[0]
    y = point[1]

    # Do this:
    x, y = point
    x, _ = point
    print(_)

# 3. Use enumerate() instead of range(len())

def demo_003():
    names = ["Alice", "Bob", "Charlie"]

    for index, name in enumerate(names, start=2):
        print(f"{index}: {name}")

#4. Dictionary Get with Defaults
def demo_004():
    user_profile = {"name": "Jay"}

    # Returns "Guest" if "role" key is missing
    role = user_profile.get("role", "Guest")
    print(role)
#5. The Walrus Operator (:=)
def demo_005():
    # Instead of checking length then using it:
    my_list = [1, 2, 3,4,5,6,7,8,9,10,11,12]
    if (n := len(my_list)) > 10:
        print(f"List is too long: {n} elements")
#6. Truthy and Falsy Evaluations
def demo_006():
    my_list = []

    if not my_list:
        print("The list is empty!")

#7. Join Strings for Efficiency
def demo_007():
    words = ["Python", "is", "awesome"]
    sentence = " ".join(words)
    print(sentence)
#8. Use zip() to Pair Iterables
def demo_008():
    keys = ["name", "age"]
    values = ["Alice", 25]

    user_dict = dict(zip(keys, values))
    print(user_dict)
#9. Generators for Memory Efficiency
def demo_009():
    # List comprehension: All 1 million items are stored in memory immediately
    big_list = [x ** 2 for x in range(1000000)]

    # Generator expression: Items are calculated one by one as you loop
    big_gen = (x ** 2 for x in range(1000000))

    print(next(big_gen))  # 0
    print(next(big_gen))  # 1

#10. Decorators to Dry Up Your Code
#Decorators allow you to "wrap" a function with another function.
# This is perfect for adding logging, timing, or authentication to many different functions
# without repeating code.
def demo_010():
    import time

    def timer_x(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"{func.__name__} took {time.time() - start:.4f}s")
            return result

        return wrapper

    def changecase(func):
        def myinner():
            return func().upper()

        return myinner

    @timer_x
    def heavy_computation():
        time.sleep(1)

    heavy_computation()

    @changecase
    def myfunction():
        return "Hello Sally"

    print(myfunction())

# 11. Lambda Functions for Quick Logic
def demo_011():
    data = [("Alice", 25), ("Bob", 20), ("Charlie", 30)]
    data.sort(key=lambda x: x[0], reverse=True)
    print(data)

# 12. Argument Unpacking with *args and **kwargs
def demo_012():
    def setup_server(host, port, debug=False):
        print(f"Connecting to {host}:{port} (Debug: {debug})")

    config = {"host": "localhost", "port": 8080, "debug": True}

    # Instead of setup_server(config['host'], config['port']...)
    setup_server(**config)

# 13. Functional Tools: map and filter
# While list comprehensions are often preferred, map and filter are highly concise
# when used with existing functions. They also return iterators (like generators),
# making them memory-efficient.
def demo_013():
    numbers = ["1", "2", "3", "4"]

    # Convert all strings to integers instantly
    ints = list(map(int, numbers))

    # Filter out numbers less than 3
    small_nums = list(filter(lambda x: x < 3, ints))
    print(small_nums)
# 14. Creating Custom Context Managers
def demo_014():
    from contextlib import contextmanager

    @contextmanager
    def simple_resource():
        print("--- Connecting to Resource ---")
        yield "The Data"
        print("--- Cleaning Up Connection ---")

    with simple_resource() as data:
        print(f"Using: {data}")

def demo_015():
    #1. Functional Dispatch Tables
    actions = {
        "add": lambda x, y: x + y,
        "power": lambda x, y: x ** y,
        "root": lambda x, y: x ** (1 / y)
    }
    # Usage
    result = actions["power"](2, 8)  # 256

    # 2. Dynamic Key Logic for Complex Sorting
    words = ["apple", "banana", "cherry", "date", "fig"]

    # Sort by length (primary), then alphabetically (secondary)
    # Returning a tuple (len(x), x) tells Python to sort by the first element, then the second
    sorted_words = sorted(words, key=lambda x: (len(x), x))

    print(sorted_words)
    from functools import reduce

    # Example: Compute the factorial of n using reduce and lambda
    n = 5
    factorial = reduce(lambda x, y: x * y, range(1, n + 1))
    print("factorial:", factorial)
    # Example: Filtering complex dictionaries
    users = [
        {"name": "Alice", "active": True},
        {"name": "Bob", "active": False},
        {"name": "Charlie", "active": True}
    ]
    active_users = list(filter(lambda u: u['active'], users))

if __name__ == '__main__':
    demo_015()
