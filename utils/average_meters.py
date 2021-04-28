'''
Used in getting the averages of training and validation
'''

class AverageMeterNorm(object):
    def __init__(self):
        self.total_num = 0
        self.total_loss = 0
        self.total_mean = 0
        self.total_median = 0
        self.total_rmse = 0
        self.total_1125 = 0
        self.total_2250 = 0
        self.total_30 = 0
        

    
    def add_batch(self, length, loss, mean, median, rmse, d_1125, d_2250, d_30):
        self.total_num += length 
        self.total_loss += length * loss 
        self.total_mean += length * mean 
        self.total_median += length * median 
        self.total_rmse += length * rmse 
        self.total_1125 += length * d_1125 
        self.total_2250 += length * d_2250 
        self.total_30 += length * d_30 
    
    def get_average(self):
        avg_loss = self.total_loss / self.total_num 
        avg_mean = self.total_mean / self.total_num
        avg_median = self.total_median / self.total_num
        avg_rmse = self.total_rmse / self.total_num
        avg_1125 = self.total_1125 / self.total_num
        avg_2250 = self.total_2250 / self.total_num
        avg_30 = self.total_30 / self.total_num
        return avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30

class AverageMeterDepth(object):
    def __init__(self):
        self.total_num = 0
        self.total_rms = 0
        self.total_rel = 0
        self.total_log10 = 0
        self.total_delta_1 = 0
        self.total_delta_2 = 0
        self.total_delta_3 = 0
        

    
    def add_batch(self, length, rms, rel, log10, delta_1, delta_2, delta_3):
        self.total_num += length 
        self.total_rms += length * rms 
        self.total_rel += length * rel
        self.total_log10 += length * log10
        self.total_delta_1 += length * delta_1
        self.total_delta_2 += length * delta_2 
        self.total_delta_3 += length * delta_3
    
    def get_average(self):
        avg_rms = self.total_rms / self.total_num
        avg_rel = self.total_rel / self.total_num
        avg_log10 = self.total_log10 / self.total_num
        avg_delta_1 = self.total_delta_1 / self.total_num
        avg_delta_2 = self.total_delta_2 / self.total_num
        avg_delta_3 = self.total_delta_3 / self.total_num
        return avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3
