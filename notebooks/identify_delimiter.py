#Test to identify delimiter

with open("data/en_ewt-ud-train.conllu",
          "r", encoding="utf-8") as file:

    for line in file:
        if line.strip() and not line.startswith("#"):
            print(repr(line))
            break