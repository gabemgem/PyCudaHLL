
def addDataCPUHLL(chll, data):
    for val in data:
        chll.add(val)

def getCardCPUHLL(chll):
    return len(chll)


def getCPUHLLCardinality(chll, data):
    addDataCPUHLL(chll, data)
    return getCardCPUHLL(chll)
