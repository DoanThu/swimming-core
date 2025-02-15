def second_to_time_str(second):
    if second < 3600:
        return "{:02}:{:.2f}".format((int(second)%3600)//60, second%60)
    return "{:02}:{:02}:{:.2f}".format(int(second)//3600, (int(second)%3600)//60, second%60)