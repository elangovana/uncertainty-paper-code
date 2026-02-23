
def expected_all_accuracy(listdict):
    num_sum = 0
    denom_sum = 0
    for t in listdict :
       if  t["p"] == "ALL": continue
       num_sum =  num_sum + t["s"]* t["p"]
       denom_sum=denom_sum+ t["s"]
    return num_sum/denom_sum

def expected_all_recall(listdict):
    num_sum = 0
    denom_sum = 0
    for t in listdict :
       if  t["p"] == "ALL": continue
       num_sum =  num_sum + t["s"]*t["p"]* t["m"]
       denom_sum=denom_sum+ t["s"]*t["m"]
    return num_sum/denom_sum

def expected_all_precision(listdict):
    num_sum = 0
    denom_sum = 0
    for t in listdict :

       if  t["p"] == "ALL": continue
       num_sum =  num_sum + t["s"]*t["p"]* t["m"]
       denom_sum = denom_sum + t["s"]*(t["p"]*t["m"]+ (1-t["p"])*(1-t["m"]))
    return num_sum/denom_sum

def expected_all_f1(listdict):

    numerator =  2* expected_all_precision(listdict)*expected_all_recall(listdict)
    denom =  expected_all_precision(listdict) + expected_all_recall(listdict)
    return numerator/denom

def expected_f1(pd, m):
    return (2 * m * pd) / (2 * m * pd + 1 - pd)

def expected_accuracy(pd, m):
    return pd

def expected_recall(pd, m):
    return pd

def expected_precision(pd, m):
    return (m*pd)/(m*pd + (1-m)* (1-pd))
