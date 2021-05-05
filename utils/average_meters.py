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
        self.total_loss = 0
        self.total_rms = 0
        self.total_rel = 0
        self.total_log10 = 0
        self.total_delta_1 = 0
        self.total_delta_2 = 0
        self.total_delta_3 = 0
        

    
    def add_batch(self, length, loss, rms, rel, rlog10, delta_1, delta_2, delta_3):
        self.total_num += length 
        self.total_loss += length * loss
        self.total_rms += length * rms 
        self.total_rel += length * rel
        self.total_log10 += length * rlog10
        self.total_delta_1 += length * delta_1
        self.total_delta_2 += length * delta_2 
        self.total_delta_3 += length * delta_3
    
    def get_average(self):
        avg_loss = self.total_loss / self.total_num
        avg_rms = self.total_rms / self.total_num
        avg_rel = self.total_rel / self.total_num
        avg_log10 = self.total_log10 / self.total_num
        avg_delta_1 = self.total_delta_1 / self.total_num
        avg_delta_2 = self.total_delta_2 / self.total_num
        avg_delta_3 = self.total_delta_3 / self.total_num
        return avg_loss, avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3

class AverageMeterSeg(object):
    def __init__(self):
        self.total_num = 0
        self.total_loss = 0
        self.total_acc = 0
    
    def add_batch(self, length, loss, acc):
        self.total_num += length 
        self.total_loss += length * loss 
        self.total_acc += length * acc
    
    def get_average(self):
        avg_loss = self.total_loss / self.total_num 
        avg_acc = self.total_acc / self.total_num
        return avg_loss, avg_acc


def get_result_string_total(the_type, epoch, epochs, batch, batchs, time, loss):
    if loss == None:
        result_string = the_type + ': Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s' \
            .format(epoch, epochs, batch, batchs, time)
    else: 
        result_string = the_type + ': Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, Loss {:.4f}' \
            .format(epoch, epochs, batch, batchs, time, loss)
    return result_string

def get_result_string_average(the_type, epoch, epochs, loss):
    if loss == None:
        result_string = the_type + ': Average: Batch: [{} / {}]' \
            .format(epoch, epochs, epoch, epochs)
    else: 
        result_string = the_type + ': Average: Batch: [{} / {}], Loss {:.4f}' \
            .format(epoch, epochs, epoch, epochs, loss)
    return result_string

def get_result_string_norm(loss, mean, median, rmse, d_1125, d_2250, d_30):
    result_string = 'loss {:.4f}, mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
        .format(loss, mean, median, rmse, d_1125, d_2250, d_30)
    return result_string
    
def get_result_string_depth(loss, rms, rel, rlog10, delta_1, delta_2, delta_3):
    result_string = 'loss {:.4f}, rms: {:.4f}, rel: {:.4f}, log10: {:.4f}, delta_1: {:.3f}, delta_2: {:.3f}, delta_3: {:.3f}' \
        .format(loss, rms, rel, rlog10, delta_1, delta_2, delta_3)
    return result_string

def get_result_string_seg(loss, acc):
    result_string = 'loss: {:.4f}, acc: {:.3f}' \
        .format(loss, acc)
    return result_string