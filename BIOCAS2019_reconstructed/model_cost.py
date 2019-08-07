import numpy as np
import lightgbm as lgb
def ReadTree(name, num_tree):
    Trees=[]
    with open(name,'r') as file:
        l=file.readline().rstrip('\n')
        for i in range(num_tree):
            tree = {}
            while not ('Tree='+str(i))==l:
                if 'end of trees' in l:
                    return Trees
                l = file.readline().rstrip('\n')
            while not 'split_feature' in l:
                l = file.readline().rstrip('\n')
            temp=l.split('=')
            split_feature=temp[1].split(' ')
            tree['split_feature']=list(map(int, split_feature))
            while not 'threshold' in l:
                l = file.readline().rstrip('\n')
            temp=l.split('=')
            threshold=temp[1].split(' ')
            tree['threshold']=list(map(float, threshold))
            while not 'left_child' in l:
                l = file.readline().rstrip('\n')
            temp=l.split('=')
            left_child=temp[1].split(' ')
            tree['left_child']=list(map(int, left_child))
            while not 'right_child' in l:
                l = file.readline().rstrip('\n')
            temp=l.split('=')
            right_child=temp[1].split(' ')
            tree['right_child']=list(map(int, right_child))
            while not 'leaf_value' in l:
                l = file.readline().rstrip('\n')
            temp=l.split('=')
            leaf_value=temp[1].split(' ')
            tree['leaf_value']=list(map(float, leaf_value))
            Trees.append(tree)
    return Trees

def one_split(tr,teX,ind, node, cost, mask):
    penalty=0
    feature_idx=tr['split_feature'][node]
    N=np.sum(mask[ind,feature_idx])
    penalty += N * cost[feature_idx]
    mask[ind,feature_idx]=False
    threshold=tr['threshold'][node]
    left_inx = teX[:, feature_idx] <= threshold
    right_inx = teX[:, feature_idx] > threshold
    left_inx = left_inx * ind
    right_inx = right_inx * ind
    if tr['left_child'][node]>=0:
        p_left=one_split(tr, teX, left_inx, tr['left_child'][node], cost, mask)
        penalty+=p_left
    if tr['right_child'][node]>=0:
        p_right=one_split(tr, teX, right_inx, tr['right_child'][node], cost, mask)
        penalty += p_right
    return penalty

def cost(teX, cost, name, num_tree):
    mask=np.ones_like(teX).astype(bool)
    Tree=ReadTree(name,num_tree)
    Total_penalty=0
    for tr in Tree:
        #mask = np.ones_like(teX).astype(bool)
        penalty=one_split(tr,teX, teX[:,0]>-float('inf'), 0, cost, mask)
        Total_penalty+=penalty
    return Total_penalty,len(mask[:,0])-np.count_nonzero(mask,0)

def size(name, num_tree):
    Tree = ReadTree(name, num_tree)
    size = 0
    for tr in Tree:
        internal=len(tr['threshold'])
        size+=internal*2+1
    return size*4/1000

def quan(line,num_bits,max_r,min_r):
    temp = line.split('=')
    leaf_value = temp[1].split(' ')
    weights = list(map(float, leaf_value))
    '''
    if max_r==None or min_r==None:
        max_r=max(weights)
        min_r=min(weights)
    elif max_r<=max(weights):
        max_r = max(weights)
    elif min_r>min(weights):
        min_r = min(weights)
    '''
    step = (max_r - min_r) / (2 ** num_bits - 1)
    #print('quantization step set to', step)
    for i in range(len(weights)):
        weights[i]=str(np.round(weights[i]/step)*step)
    l=' '.join(weights)
    return 'leaf_value='+l+'\n'

def change_size(line,model_size):
    temp = line.split('=')
    leaf_value = temp[1].split(' ')
    weights = list(map(int, leaf_value))
    for i in range(len(weights)):
        weights[i]=str(np.round(weights[i]+model_size[i]))
    l = ' '.join(weights)
    return 'tree_sizes=' + l + '\n'

def quantization(num_bits,name='model.txt'):
    from tempfile import mkstemp
    from shutil import move
    from os import fdopen, remove
    if num_bits==0:
        from shutil import copyfile
        copyfile('model.txt', 'quan_model.txt')
        return 0
    Tree = ReadTree('model.txt', 100)
    max_r = float('-inf')
    min_r = float('inf')
    for t in Tree:
        max_r=max(max_r,max(t['leaf_value']))
        min_r = min(min_r, min(t['leaf_value']))
    step = (max_r - min_r) / (2 ** num_bits - 1)
    #print('quantization step set to', step)
    model_size = np.ones(len(Tree),dtype='int')*16
    tree_ind=0
    fh_t, abs_path_t = mkstemp()
    with fdopen(fh_t, 'w') as new_file:
        with open('model.txt') as old_file:
            for line in old_file:
                if not 'leaf_value' in line:
                    new_file.write(line)
                else:
                    l=quan(line,num_bits,max_r,min_r)
                    new_file.write(l)
                    model_size[tree_ind] += len(l)-len(line)
                    tree_ind+=1
    move(abs_path_t, 'quan_model.txt')
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open('quan_model.txt') as old_file:
            for line in old_file:
                if not 'tree_sizes=' in line:
                    new_file.write(line)
                else:
                    l=change_size(line,model_size)
                    new_file.write(l)
    move(abs_path, 'quan_model.txt')

    return step

def get_leaf_weights(name):
    Tree = ReadTree(name, 100)
    weights=[]
    for t in Tree:
        weights=weights+t['leaf_value']
    return weights
