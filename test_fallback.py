def generate_fallback(candidates, like_budget, comment_budget):
    selected = []
    engage_count = 0
    like_count = 0
    for c in candidates:
        if engage_count < comment_budget and (engage_count + like_count) < like_budget:
            selected.append(dict(c, action="ENGAGE"))
            engage_count += 1
        elif (engage_count + like_count) < like_budget:
            selected.append(dict(c, action="LIKE"))
            like_count += 1
        else:
            break
    return selected

print(generate_fallback([{"id": i} for i in range(10)], 5, 2))
