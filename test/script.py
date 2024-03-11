from radon.complexity import cc_rank, cc_visit

with open('/Users/shahriyar/Desktop/programming/Python/Specefic_Group/Work/Customer_Churn_Classification-main/mian.py') as f:
    code = f.read()

complexity_results = cc_visit(code)
# cc_complexity=cc_rank(code)
for result in complexity_results:
# for result in complexity_results:
    print(result)