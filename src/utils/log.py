def log(msg, type):
    if type == "error":
        print("\033[1m\033[91mERROR\033[0m", end=" ")

    print(msg)
