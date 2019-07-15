def addLog(req_date, req_ip, text, prediction, score):
    c = open("request_logger.txt", "a")
    c.write(f"{req_date}\t{req_ip}\t{text}\t{prediction}\t{score}\n")
    
    count = getRequestNumber()
    return count

def getRequestNumber():
    c = open("request_logger.txt", "r")
    c = [x.rstrip("\n") for i, x in enumerate(c) if i != 0]
    return len(c)