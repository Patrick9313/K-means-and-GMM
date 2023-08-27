def find_possible_lengths(s):
    n = len(s)
    lengths = set()

    for i in range(1, n):
        if n % i == 0:
            substring = s[:i]
            if substring * (n // i) == s:
                lengths.add(i)

    lengths.add(n)  # 将字符串长度本身也添加到结果中

    return sorted(lengths)

# 接收用户输入
s = input()
possible_lengths = find_possible_lengths(s)
print(len(possible_lengths))
print(" ".join(map(str, possible_lengths)))
