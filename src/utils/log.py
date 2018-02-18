def log(msg, type="normal"):
    if type == "error":
        print("\033[1m\033[91mERROR\033[0m", end=" ")
    elif type == "normal":
        print("\033[1m\033[94mLOG  \033[0m", end=" ")
    print(msg)
