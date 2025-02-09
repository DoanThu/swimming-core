def second_to_time_str(second):
    return "{}:{}:{:.2f}".format(second//3600, (second%3600)//60, second%60)