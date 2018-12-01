
import random
import math
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
random.seed(3428)

#random.seed(1234)
############# ############# ############# ############# #############  HELPER FUNCTIONS ############# ############# ############# ############# ############# 
def add_stumps(x1, x2, x3, x4):
    stumps = []
    if x2 - x1 < x3 - x2 and x2 - x1 < x4 - x3:  # R|_|_
        stumps.append([x2, x3, 0])
        stumps.append([x2, x3, 1])
    if x3 - x2 < x2 - x1 and x3 - x2 < x4 - x3:  # _|R|_
        stumps.append([x2, x3, 4])
        stumps.append([x2, x3, 5])
    if x4 - x3 < x2 - x1 and x4 - x3 < x3 - x2:  # _|_|R
        stumps.append([x2, x3, 2])
        stumps.append([x2, x3, 3])
    return stumps

def create_experts(K, want_random,one_d,R):

    if one_d==0:
         experts=[]
         actual_K=2*(K-1)
         print 'real K'
         print actual_K
         update_matrix=np.identity(actual_K)
         interval= np.linspace(-1.0, 1.0, K)
         count_K=0
         for itr in range(len(interval)-1): 
             if itr == 0:
                 experts.append([interval[itr],interval[itr+1],0]  )
                 experts.append([interval[itr],interval[itr+1],1]  )
             elif 0 < itr < len(interval)-2:
                 experts.append([interval[itr],interval[itr+1],4]  )
                 experts.append([interval[itr],interval[itr+1],5]  )

             else:
                 experts.append([interval[itr],interval[itr+1],2]  )
                 experts.append([interval[itr],interval[itr+1],3]  )

             update_matrix[count_K,count_K+1]=1.
             update_matrix[count_K+1,count_K]=1.
             count_K+=2
         regional_experts=[]
#         enumerate_experts=list(range(len(experts)))

#         for itr in range(0,2*len(interval)-2,2): 
#             regional_experts.append([x for x in enumerate_experts if x != itr and x!=itr+1])


    elif one_d==1:

        hyp_experts = list(np.linspace(0, np.pi, K/float(R)))
###
#        temp_hyp_experts= list(np.linspace(0, np.pi, K/float(R)))
#        for j in temp_hyp_experts:
#            if  j < (1/4.)*np.pi + 0.1*2*np.pi and j > (1/4.)*np.pi - 0.1*2*np.pi:
#             hyp_experts.remove(j)
####

        rej_experts = list(np.linspace(0, 0.99, R))
        experts=[]

#        count_K=0

        for hyp in hyp_experts:
#            rej_index=0
            for rej in rej_experts:
                experts.append([hyp,rej])
#                for i in range(rej_index):
#                    update_matrix[count_K,count_K-(i+1)]=1.
#                rej_index+=1
#                count_K+=1
        K=len(experts)
        update_matrix=np.identity(K)
        for i in range(0,K,R):
            for j in range(0,K,R):
                cur=np.identity(R)
                cur[0,:]=np.ones(R)
                update_matrix[i:i+R,j:j+R]=cur
#                update_matrix[i:i+R,j:j+R]=np.triu(np.ones((R,R)))

        regional_experts=[]
#        for i in range(R):
#            current=[]
#            for j in range(R):
#                if j<=i:
#                    current.extend(range(j,K,R))
#            regional_experts.append(current)
#        print 'real K'
#        print update_matrix.shape

#        print experts
#        print regional_experts
#        print update_matrix[0:20,0:20]
#        print np.array(update_matrix).shape
    elif one_d==2: 

        hyp_experts = list(np.linspace(0.1, 0.5*np.pi, K / float(R)))
        hyp_experts2 = list(np.linspace(0.1, 0.5*np.pi, K / float(R)))
        rej_experts = list(np.linspace(0.0, 0.99, R))
        experts=[]

#        count_K=0
        for hyp in hyp_experts:
            for hyp2 in hyp_experts2:
#                rej_index=0
                for rej in rej_experts:
                    experts.append([hyp,hyp2,rej])
#                    for i in range(rej_index):
#                        update_matrix[count_K,count_K-(i+1)]=1.
#                    rej_index+=1
#                    count_K+=1
        real_K= R* len(hyp_experts)* len(hyp_experts2)
        print "real K "
        print len(experts)
        update_matrix=np.identity(real_K)
        for i in range(0,real_K,R):
            for j in range(0,real_K,R):
                cur=np.identity(R)
                cur[0,:]=np.ones(R)
                update_matrix[i:i+R,j:j+R]=cur
                #update_matrix[i:i+R,j:j+R]=np.triu(np.ones((R,R)))

        regional_experts=[]
#        for i in range(R):
#            current=[]
#            for j in range(R):
#                if j<=i:
#                    current.extend(range(j,K,R))
#            regional_experts.append(current)

    else:
#        random.seed(52)
        hyp_experts=[]
        for erp in range(K):
            hyp_experts.append(np.random.normal(0, 1, one_d) )


        rej_experts = list(np.linspace(0.1, math.sqrt(one_d), R))
        experts=[]
        for hyp in hyp_experts:
            for rej in rej_experts:
                experts.append([hyp,rej])
        real_K= R * K
        print len(experts)
        update_matrix=np.identity(real_K)
        for i in range(0,real_K,R):
            for j in range(0,real_K,R):
                cur=np.identity(R)
#                cur[0,:]=np.ones(R)
                update_matrix[i:i+R,j:j+R]=cur
        regional_experts=[]
 #       for i in range(R):
 #           current=[]
 #           for j in range(R):
 #               if j<=i:
 #                   current.extend(range(j,K,R))
 #           regional_experts.append(current)

    if len(experts)!= len(update_matrix[0,:]):
        print "ERROR: expert length is different than update_matrix column length"
        print len(experts)
        print len(update_matrix[0,:])
    return experts,update_matrix,regional_experts

def create_data(T,type_data):
 
    #creating data according to gaussian and labels
    if type_data==0:
        x_data = [random.uniform(-1.0, 1.0) for itr in range(T)]
        y_labels = [ int(itr >= 0.5)  for itr in x_data]
#    x_data = [random.uniform(0, 1) for itr in range(T)]
        data = zip(x_data, y_labels)  #data format is list of tuple [(x1,y1),(x2,y2)....]
    elif type_data==1:
        x_data = [[random.uniform(-1.0, 1.0),random.uniform(-1.0,1.0)] for itr in range(T)]
        y_labels = [ int(itr[0] + itr[1] > 0)  for itr in x_data] #label +1 if w_1 x_1 + w_2 x_2 > 0
        data = zip(x_data, y_labels)  #data format is list of tuple [(x1,y1),(x2,y2)....]

    elif type_data==2:
        #load data
        cifar_data = np.genfromtxt('cifar10pca.txt', delimiter=',')
        x_data_comp1=cifar_data[:,0] #first component
        #scale data by max
        x_data=x_data_comp1/float(np.amax(np.absolute(x_data_comp1)))
        cifar_label = np.genfromtxt('cifar10labels.txt', delimiter=',')
        y_labels=[]
        for lab in cifar_label:
            if int(lab)==7:
                y_labels.append(0)
            else:
                y_labels.append(1)
        data = zip(x_data, y_labels)  
        np.random.shuffle(data) #shuffle data
        data=data[:T]

    elif type_data==3:
        new_data = np.genfromtxt('skin.txt', delimiter='\t')
#        x_data_comp1=cifar_data[:,0] #first component
        #scale data by max
        x_data=new_data[:,0:3]
        x_data[:,0]=x_data[:,0]/float(np.amax(np.absolute(x_data[:,0])))
        x_data[:,1]=x_data[:,1]/float(np.amax(np.absolute(x_data[:,1])))
        x_data[:,2]=x_data[:,2]/float(np.amax(np.absolute(x_data[:,2])))

        new_label =new_data[:,-1]


        y_labels=[]
        for lab in new_label:
            if int(lab)==2:
                y_labels.append(0)
            else:
                y_labels.append(1)
        data = zip(x_data.tolist(), y_labels)  
        np.random.shuffle(data) #shuffle data
        data=data[:T]

    elif type_data==4:
        new_data = np.genfromtxt('guide', delimiter=',')
        x_data=new_data[:,1:]
        for col in range(len(x_data[0,:])):
            x_data[:,col]=x_data[:,col]/float(np.amax(np.absolute(x_data[:,col])))
        y_data=new_data[:,0]
        y_labels=[]
        for lab in y_data:
                y_labels.append(lab)
        data = zip(x_data.tolist(), y_labels)
        np.random.shuffle(data) #shuffle data                                                                                                                                                                       
        data=data[:T]

    elif type_data==8:
        new_data = np.genfromtxt('cod.txt', delimiter=',')
        x_data=new_data[:,1:]
        for col in range(len(x_data[0,:])):
            x_data[:,col]=x_data[:,col]/float(np.amax(np.absolute(x_data[:,col])))
        y_data=new_data[:,0]
        y_labels=[]
        for lab in y_data:
            if int(lab)==-1:
                y_labels.append(0)
            else:
                y_labels.append(1)
        data = zip(x_data.tolist(), y_labels)
        np.random.shuffle(data) #shuffle data                                                                                                                                                                       
        data=data[:T]

    elif type_data==14:
        new_data = np.genfromtxt('eye.txt', delimiter=',')
        x_data=new_data[:,0:-1]
        for col in range(len(x_data[0,:])):
            x_data[:,col]=x_data[:,col]/float(np.amax(np.absolute(x_data[:,col])))
        y_data=new_data[:,-1]
        y_labels=[]
        for lab in y_data:
            y_labels.append(lab)
        data = zip(x_data.tolist(), y_labels)
        np.random.shuffle(data) #shuffle data                                                                                                                                                                       
        data=data[:T]




    elif type_data==22:
        new_data = np.genfromtxt('ijcnn.txt', delimiter=',')
        x_data=new_data[:,1:]
        for col in range(len(x_data[0,:])):
            x_data[:,col]=x_data[:,col]/float(np.amax(np.absolute(x_data[:,col])))
        y_data=new_data[:,0]
        y_labels=[]
        for lab in y_data:
            if int(lab)==-1:
                y_labels.append(0)
            else:
                y_labels.append(1)


        data = zip(x_data.tolist(), y_labels)
        np.random.shuffle(data) #shuffle data                                                                                                                                                                       
        data=data[:T]


    elif type_data==25:
        cifar_data = np.genfromtxt('cifar10pca.txt', delimiter=',')
        x_data_comp1=cifar_data[:,:25] #all components
        #scale data by max
        x_data=x_data_comp1/float(np.amax(np.absolute(x_data_comp1)))
        cifar_label = np.genfromtxt('cifar10labels.txt', delimiter=',')
        y_labels=[]
        for lab in cifar_label:
            if int(lab)==7:
                y_labels.append(0)
            else:
                y_labels.append(1)
        data = zip(x_data, y_labels)  
        np.random.shuffle(data) #shuffle data
        data=data[:T] 

    elif type_data==28:
        new_data = np.genfromtxt('higgs', delimiter=',')
        x_data=new_data[:,1:]
        for col in range(len(x_data[0,:])):
            x_data[:,col]=x_data[:,col]/float(np.amax(np.absolute(x_data[:,col])))
        y_data=new_data[:,0]
        y_labels=[]
        for lab in y_data:
            y_labels.append(lab)

        data = zip(x_data.tolist(), y_labels)
        np.random.shuffle(data) #shuffle data                                                                                                                                                                       
        data=data[:T]



    elif type_data==54:
        new_data = np.genfromtxt('covtype', delimiter=',')
        x_data=new_data[:,1:]
#        for col in range(len(x_data[0,:])):
#            x_data[:,col]=x_data[:,col]/float(np.amax(np.absolute(x_data[:,col])))
        y_data=new_data[:,0]
        y_labels=[]
        for lab in y_data:
            if int(lab)==2:
                y_labels.append(0)
            else:
                y_labels.append(1)

        data = zip(x_data.tolist(), y_labels)
        np.random.shuffle(data) #shuffle data                                                                                                                                                                       
        data=data[:T]


    elif type_data==68:
        new_data = np.genfromtxt('phish', delimiter=',')
        x_data=new_data[:,1:]
        for col in range(len(x_data[0,:])):
            x_data[:,col]=x_data[:,col]/float(np.amax(np.absolute(x_data[:,col])))
        y_data=new_data[:,0]
        y_labels=[]
        for lab in y_data:
            y_labels.append(lab)

        data = zip(x_data.tolist(), y_labels)
        np.random.shuffle(data) #shuffle data                                                                                                                                                                       
        data=data[:T]



    elif type_data==123:
        new_data = np.genfromtxt('a9a', delimiter=',')
        x_data=new_data[:,1:]
        for col in range(len(x_data[0,:])):
            x_data[:,col]=x_data[:,col]/float(np.amax(np.absolute(x_data[:,col])))
        y_data=new_data[:,0]
        y_labels=[]
        for lab in y_data:
            y_labels.append(int(lab))

        data = zip(x_data.tolist(), y_labels)
        np.random.shuffle(data) #shuffle data                                                                                                                                                                       
        data=data[:T]
     
    return data


def lcb_bound(current_time, pull, alpha):
    if pull == 0:
        return float("inf")
    else:
        return math.sqrt((alpha / 2)* math.log(current_time) / float(pull))

def loss_of_best_expert(dat,experts,c):
    return min(loss_of_every_expert(dat, experts, c))

def loss_of_every_expert(dat, experts, c,return_rounds,one_d,R):
    loss_expert_at_rounds=[]
    num_reject_at_rounds=[]
    enum_return_rounds=0
    loss_expert = [0] * len(experts)
    num_reject= [0] * len(experts)
    for t in range(len(dat)):
        for i in range(len(experts)):

                loss_expert[i] += rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d,R), c)
                if rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d,R), c)==c:
                    num_reject[i]+=1
        if enum_return_rounds < len(return_rounds) and t+1 ==return_rounds[enum_return_rounds]:
            loss_expert_at_rounds.append([ix / float(t+1) for ix in loss_expert] )
            num_reject_at_rounds.append([ix / float(t+1) for ix in num_reject] )
            enum_return_rounds+=1

    return loss_expert_at_rounds,num_reject_at_rounds

def loss_of_given_expert(dat, experts, c,return_rounds,one_d,R,given):
    loss_expert_at_rounds=[]

    enum_return_rounds=0
    loss_expert = 0
    num_reject=0
    for t in range(len(dat)):

        loss_expert += rej_loss(dat[t][1], exp_label(dat[t][0], experts[given],one_d,R), c)
        if rej_loss(dat[t][1], exp_label(dat[t][0], experts[given],one_d,R), c)    == c:
            num_reject+=1

        if enum_return_rounds < len(return_rounds) and t+1 ==return_rounds[enum_return_rounds]:
            loss_expert_at_rounds.append([ix / float(t+1) for ix in loss_expert] )
            num_reject_at_rounds.append([ix / float(t+1) for ix in num_reject] )
            enum_return_rounds+=1
    
    return loss_expert_at_rounds,num_reject_at_rounds




def rej_loss(true_label, expert_label, c):

    if expert_label == -1:
        return c
    else:
        if true_label == expert_label:
            return 0.0
        else:
            return 1.0

def acc_loss(true_label, expert_label):
    if expert_label==-1:
        print "ERROR IN ACC_LOSS ExPERT LABEL"

    if true_label == expert_label:
        return 0.0
    else:
        return 1.0

def exp_hyp_label(data, expert,one_d):
    if one_d==0:
        if expert[2]==0 or expert[2]==4:
            if expert[1] <= data: 
                expert_label = 1
            else:
                expert_label = 0
        elif expert[2]==1:
            if expert[1] >= data:
                expert_label = 1
            else:
                expert_label = 0
        elif expert[2]==2:
            if expert[0] <= data:
                expert_label = 1
            else:
                expert_label = 0
        elif expert[2]==3  or expert[2]==5:
            if expert[0] >= data:
                expert_label = 1
            else:
                expert_label = 0
        else:
            print 'incorrect stump type'

    elif one_d==1:
        if np.cos(expert[0]) * data[0] + np.sin(expert[0]) * data[1] > 0:
            expert_label = 1
        else:
            expert_label = 0
    elif one_d==2:
        if np.sin(expert[1])*np.cos(expert[0]) * data[0] +np.sin(expert[1])* np.sin(expert[0]) * data[1]- np.cos(expert[1])*data[2] > 0:
            expert_label = 1
        else:
            expert_label = 0
    else:

       if np.dot(expert[0], data) > 0:
           expert_label = 1
       else:
           expert_label = 0

    return expert_label



  
def exp_label(data, expert,one_d,R):
    if one_d==0:
        if expert[2]==0 or expert[2]==1: #R|0|1 and R|1|0
            if expert[0] < data:
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
               expert_label = -1
        elif expert[2]==2 or expert[2]==3: #0|1|R and 1|0|R 
            if expert[1] > data:
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
               expert_label = -1
        elif expert[2]==4 or expert[2]==5:  #0|R|1 and 1|R|0
            if expert[0]< data < expert[1]:
                expert_label = -1
            else:
                expert_label = exp_hyp_label(data, expert,one_d)

        else:
                print 'incorrect stump type'

    elif one_d==1:
        rej_experts = list(np.linspace(0.0, 0.99, R))
        
        if expert[1]==rej_experts[1]:
            if LA.norm(data) >= expert[1]:
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
                expert_label = -1

        elif expert[1]!=0:
            if  LA.norm(data) >= expert[1] or LA.norm(data) < rej_experts[rej_experts.index(expert[1])-1]  : # this is when you accept. if distance to plane is higher than threhsodl
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
                expert_label = -1
        
        else:
                expert_label = exp_hyp_label(data, expert,one_d)


    elif one_d==2:
        rej_experts = list(np.linspace(0.0, 0.99, R))
        if expert[2]==rej_experts[1]:
            if LA.norm(data) >= expert[2]:
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
                expert_label = -1

        elif expert[2]!=0:
            if  LA.norm(data) >= expert[2] or LA.norm(data) < rej_experts[rej_experts.index(expert[2])-1]  : # this is when you accept. if distance to plane is higher than threhsodl
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
                expert_label = -1
        
        else:
                expert_label = exp_hyp_label(data, expert,one_d)

    else:


        rej_experts = list(np.linspace(0.1, math.sqrt(one_d), R))
        if expert[1]==rej_experts[0]:
            if LA.norm(data) >= expert[1]:
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
                expert_label = -1

        else:
            if  LA.norm(data) >= expert[1] or LA.norm(data) < rej_experts[rej_experts.index(expert[1])-1]  : # this is when you accept. if distance to plane is higher than threhsodl
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
                expert_label = -1


        # rej_experts = list(np.linspace(0.05, 0.99, R))
        # if expert[1]==rej_experts[1]:
        #     if LA.norm(data) >= expert[1]:
        #         expert_label = exp_hyp_label(data, expert,one_d)
        #     else:
        #         expert_label = -1

        # elif expert[1]!=0:
        #     if  LA.norm(data) >= expert[1] or LA.norm(data) < rej_experts[rej_experts.index(expert[1])-1]  : # this is when you accept. if distance to plane is higher than threhsodl
        #         expert_label = exp_hyp_label(data, expert,one_d)
        #     else:
        #         expert_label = -1
        
        # else:
        #         expert_label = exp_hyp_label(data, expert,one_d)

    return expert_label


def exp_reject(data, expert,one_d,R):
    if one_d==0:
        if expert[2]==0 or expert[2]==1: #R|0|1 and R|1|0
            if expert[0] < data:
                expert_label = 0
            else:
               expert_label = -1
        elif expert[2]==2 or expert[2]==3: #0|1|R and 1|0|R 
            if expert[1] > data:
                expert_label = 0
            else:
               expert_label = -1
        elif expert[2]==4 or expert[2]==5:  #0|R|1 and 1|R|0
            if expert[0]< data < expert[1]:
                expert_label = -1
            else:
                expert_label = 0

        else:
                print 'incorrect stump type'

    elif one_d==1:
        rej_experts = list(np.linspace(0.0, 0.99, R))
        
        if expert[1]==rej_experts[1]:
            if LA.norm(data) >= expert[1]:
                expert_label = 0
                
            else:
                expert_label = -1

        elif expert[1]!=0:
            if  LA.norm(data) >= expert[1] or LA.norm(data) < rej_experts[rej_experts.index(expert[1])-1]  : # this is when you accept. if distance to plane is higher than threhsodl
                expert_label = 0
            else:
                expert_label = -1
        
        else:
                expert_label = 0
    elif one_d==2:
        rej_experts = list(np.linspace(0.0, 0.99, R))
        if expert[2]==rej_experts[1]:
            if LA.norm(data) >= expert[2]:
                expert_label = 0
            else:
                expert_label = -1

        elif expert[2]!=0:
            if  LA.norm(data) >= expert[2] or LA.norm(data) < rej_experts[rej_experts.index(expert[2])-1]  : # this is when you accept. if distance to plane is higher than threhsodl
                expert_label = 0
            else:
                expert_label = -1
        
        else:
                expert_label = 0
      
    else:

        rej_experts = list(np.linspace(0.1, math.sqrt(one_d), R))
        if expert[1]==rej_experts[0]:
            if LA.norm(data) >= expert[1]:
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
                expert_label = -1

        else:
            if  LA.norm(data) >= expert[1] or LA.norm(data) < rej_experts[rej_experts.index(expert[1])-1]  : # this is when you accept. if distance to plane is higher than threhsodl
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
                expert_label = -1



        # rej_experts = list(np.linspace(0.0, 0.99, R))
        # if expert[1]==rej_experts[1]:
        #     if LA.norm(data) >= expert[1]:
        #         expert_label = 0
        #     else:
        #         expert_label = -1

        # elif expert[1]!=0:
        #     if  LA.norm(data) >= expert[1] or LA.norm(data) < rej_experts[rej_experts.index(expert[1])-1]  : # this is when you accept. if distance to plane is higher than threhsodl
        #         expert_label = 0
        #     else:
        #         expert_label = -1
        
        # else:
        #         expert_label = 0

    return expert_label

  

def dist_to_plane(data,expert):
    # return math.fabs(expert[0]*data[0]-data[1]+0.5)/math.sqrt(expert[0]**2+1+0.5**2)
    return math.fabs(np.cos(expert[0]) * data[0] + np.sin(expert[0]) * data[1])/math.sqrt(np.cos(expert[0])**2+np.sin(expert[0])**2)

def dist_to_plane2(data,expert):
    # return math.fabs(expert[0]*data[0]-data[1]+0.5)/math.sqrt(expert[0]**2+1+0.5**2)
    return math.fabs(np.sin(expert[1])*np.cos(expert[0]) * data[0] + np.sin(expert[1])*np.sin(expert[0]) * data[1]- np.cos(expert[1])*data[2] )/math.sqrt((np.sin(expert[1])*np.cos(expert[0]))**2+(np.sin(expert[1])*np.sin(expert[0]))**2+ np.cos(expert[1])**2)



############# ############# ############# ############# ALGORITHMS ############# ############# ############# ############# ############# ############# 



def ucb(c, alpha, experts, dat, return_rounds, one_d,update_matrix,R):
    save_arms=[]
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    edges_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n UCB N'
    expert_avg = [0.0]*K
    expert_pulls = [0.0] * K 
    loss_alg = 0
    count_rej=0

    total_edges=0
    for t in range(T):

        #find best arm
        lcb_list=[max(expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha), 0.0) for i in range(K)] #soinefficient

        cur_min=min(lcb_list)
        min_index=[i for i, v in enumerate(lcb_list) if v == cur_min]
        best_arm = random.choice(min_index)#expert_lcbs.index(min(expert_lcbs)) 
#        best_arm = lcb_list.index(min(lcb_list)) 
        save_arms.append(best_arm)

        #update regret
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d,R)
        cur_loss=rej_loss(dat[t][1], expert_label, c) 
        loss_alg +=cur_loss# rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
        #        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbN'
        #        print best_arm,expert_loss

        # all experts whose acceptance region is a subset of accetange region of best arm. 
#        current_update=update_matrix[best_arm,:]
#        for i in range(K):
#            if current_update[i] == 1.: 
        expert_pulls[best_arm] += 1
        inv_pull = 1.0 / expert_pulls[best_arm]
        expert_avg[best_arm] = cur_loss * inv_pull + (1 - inv_pull) * expert_avg[best_arm]
                
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            edges_at_return_rounds.append(total_edges)
            enum_return_rounds+=1

    return loss_alg_at_return_rounds,count_rej_at_return_rounds,save_arms,edges_at_return_rounds


def ftl(c, alpha, experts, dat, return_rounds, one_d,update_matrix,R):
    save_arms=[]
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    edges_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n UCB N'
    expert_avg = [0.0]*K
    expert_pulls = [0.0] * K 
    loss_alg = 0
    count_rej=0

#    total_edges=np.count_nonzero(update_matrix)
    total_edges=K*K
    for t in range(T):

        #find best arm
        lcb_list=[max(expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha), 0.0) for i in range(K)] #soinefficient

        cur_min=min(lcb_list)
        min_index=[i for i, v in enumerate(lcb_list) if v == cur_min]
        best_arm = random.choice(min_index)#expert_lcbs.index(min(expert_lcbs)) 
#        best_arm = lcb_list.index(min(lcb_list)) 
        save_arms.append(best_arm)

        #update regret
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d,R)
        cur_loss=rej_loss(dat[t][1], expert_label, c) 
        loss_alg +=cur_loss# rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
        #        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbN'
        #        print best_arm,expert_loss

        # all experts whose acceptance region is a subset of accetange region of best arm. 
#        current_update=update_matrix[best_arm,:]
#        for i in range(K):
#            if current_update[i] == 1.: 
        for i in range(K):
            expert_pulls[i] += 1
            inv_pull = 1.0 / expert_pulls[i]
            expert_avg[i] = rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d,R), c)  * inv_pull + (1 - inv_pull) * expert_avg[i]

                
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            edges_at_return_rounds.append(total_edges)
            enum_return_rounds+=1

    return loss_alg_at_return_rounds,count_rej_at_return_rounds,save_arms,edges_at_return_rounds


def ucboldn(c, alpha, experts, dat, return_rounds, one_d,update_matrix,R):
    save_arms=[]
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    edges_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n UCB N'
    expert_avg = [0.0]*K
    expert_pulls = [0.0] * K 
    loss_alg = 0
    count_rej=0

#    total_edges=np.count_nonzero(update_matrix)
    for t in range(T):

        #find best arm
        lcb_list=[max(expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha), 0.0) for i in range(K)] #soinefficient

        cur_min=min(lcb_list)
        min_index=[i for i, v in enumerate(lcb_list) if v == cur_min]
        best_arm = random.choice(min_index)#expert_lcbs.index(min(expert_lcbs)) 
#        best_arm = lcb_list.index(min(lcb_list)) 
        save_arms.append(best_arm)

        #update regret

        expert_label = exp_label(dat[t][0], experts[best_arm],one_d,R)
        cur_loss=rej_loss(dat[t][1], expert_label, c) 
        loss_alg +=cur_loss# rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
            #update only rejecting experts since "never" receive true label
            current_rej=0
            for i in range(K): #soinefficient
                if exp_label(dat[t][0], experts[i],one_d,R) == -1:
                    expert_pulls[i] += 1
                    inv_pull = 1.0 / expert_pulls[i]
                    expert_avg[i] = c * inv_pull + (1 - inv_pull) * expert_avg[i]
                    current_rej+=1
            total_edges=current_rej
        else:
            #update all experts since received true label. 
            for i in range(K): 
                expert_pulls[i] += 1
                inv_pull = 1.0 / expert_pulls[i]
                expert_avg[i] = rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d,R), c)  * inv_pull + (1 - inv_pull) * expert_avg[i]
            total_edges=K*K

            
                
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            edges_at_return_rounds.append(total_edges)
            enum_return_rounds+=1

    return loss_alg_at_return_rounds,count_rej_at_return_rounds,save_arms,edges_at_return_rounds








def ucbn(c, alpha, experts, dat, return_rounds, one_d,update_matrix,R):
    save_arms=[]
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    edges_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n UCB N'
    expert_avg = [0.0]*K
    expert_pulls = [0.0] * K 
    loss_alg = 0
    count_rej=0

    total_edges=np.count_nonzero(update_matrix)
    for t in range(T):

        #find best arm
        lcb_list=[max(expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha), 0.0) for i in range(K)] #soinefficient

        cur_min=min(lcb_list)
        min_index=[i for i, v in enumerate(lcb_list) if v == cur_min]
        best_arm = random.choice(min_index)#expert_lcbs.index(min(expert_lcbs)) 
#        best_arm = lcb_list.index(min(lcb_list)) 
        save_arms.append(best_arm)

        #update regret
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d,R)
        loss_alg += rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
        #        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbN'
        #        print best_arm,expert_loss

        # all experts whose acceptance region is a subset of accetange region of best arm. 
        current_update=update_matrix[best_arm,:]
        for i in range(K):
            if current_update[i] == 1.: 
                expert_pulls[i] += 1
                inv_pull = 1.0 / expert_pulls[i]
                expert_avg[i] = rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d,R), c)  * inv_pull + (1 - inv_pull) * expert_avg[i]
                
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            edges_at_return_rounds.append(total_edges)
            enum_return_rounds+=1

    return loss_alg_at_return_rounds,count_rej_at_return_rounds,save_arms,edges_at_return_rounds

def ucbb(c, alpha, experts, dat, return_rounds, one_d,update_matrix,R):
    save_arms=[]
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    edges_at_return_rounds=[]
    enum_return_rounds=0
    K = len(experts)
    T = len(dat)
#    print '\n\n UCB N'
    expert_avg = [0.0]*K
    expert_pulls = [0.0] * K 
    loss_alg = 0
    count_rej=0


    prob_region=np.zeros((R,R))
    threshold=np.ones(K)
    
    
    for t in range(T):
         

        #find best arm
        lcb_list=[max(expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha), 0.0) for i in range(K)] #soinefficient

        cur_min=min(lcb_list)
        min_index=[i for i, v in enumerate(lcb_list) if v == cur_min]
        best_arm = random.choice(min_index)
        save_arms.append(best_arm)

        #update regret
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d,R)
        loss_alg += rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1

        # all experts whose acceptance region is a subset of accetange region of best arm.
        current_update=prob_region[best_arm%R,:]

        for i in range(K):
            if current_update[i%R] <= threshold[i]: 
                expert_pulls[i] += 1
                inv_pull = 1.0 / expert_pulls[i]
                expert_i_label=exp_label(dat[t][0], experts[i],one_d,R)
                if expert_label == -1 and expert_i_label  != -1:  #if chosen expert reject but i accepts. 
                  expert_avg[i] =  (1 - inv_pull) * expert_avg[i]
                else:
                  expert_avg[i] = rej_loss(dat[t][1], expert_i_label, c)  * inv_pull + (1 - inv_pull) * expert_avg[i]

            # update the threshold and prob_region

            threshold[i]=math.sqrt(expert_pulls[i] * math.log((alpha/2)*(t+1)))/( float((K-1.)*(t+1)))
#            if expert_i_label !=-1:
#              for j in range(R):
#                if exp_reject(dat[t][0], experts[j],one_d,R) == -1:  #if expert j reject but i accepts. 
#                  prob_region[j,i]= (prob_region[j,i]*(t-1.)+1.) / float(t)

        for k in range(R):
          for n in range(R):
             if exp_reject(dat[t][0], experts[k],one_d,R) == -1 and  exp_reject(dat[t][0], experts[n],one_d,R) !=-1:
               prob_region[k,n]=(prob_region[k,n]*t+1.)/float(t+1)
             else:
               prob_region[k,n]=(prob_region[k,n]*t)/float(t+1)
#        for i in range(0,K,R):
#          for j in range(0,K,R):
#                prob_region[i:i+R,j:j+R]=small_prob

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:


            total_edges=0
            for exr in range(K):
                thr=threshold[exr]
                cur_row=prob_region[exr%R]
                for val in cur_row:
                    if val <= thr:
                        total_edges+=1

            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            edges_at_return_rounds.append(total_edges*(K/float(R)))
            enum_return_rounds+=1

    return loss_alg_at_return_rounds,count_rej_at_return_rounds,save_arms,edges_at_return_rounds


def get_region_id(data,regional_experts,one_d,K,R):
    if one_d==0:
        interval= np.linspace(-1.0, 1.0, K)
        return np.where(np.sort(np.append(interval,data))==data)[0][0]-1
    elif one_d>=1:
        rej_experts = list(np.linspace(0.0, 0.99, R))
        for rej in rej_experts:
            if LA.norm(data)< rej:
                return rej_experts.index(rej)-1
                break
            elif LA.norm(data) >= rej_experts[-1]:
                return R-1
                break


def ucbh(c, alpha, experts, dat, return_rounds, one_d,update_matrix,K,regional_experts,R):
    save_arms=[]
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0
    loss_alg = 0
    count_rej=0
    T = len(dat)
#    print('regionalexperts:')
#    print(regional_experts)
    num_regional_experts=len(regional_experts)
    number_pulls=[0.0]*num_regional_experts
    awake_experts_avg=[]
    for reg_id in range(num_regional_experts):
        awake_experts_avg.append([0.0]*len(regional_experts[reg_id]) )

    for t in range(T):

        region_id= get_region_id(dat[t][0],regional_experts,one_d,K,R)
 #       print('data and region id')
 #       print(dat[t][0])
#        print(regional_experts)
#        print(K)
#        print(region_id)
        
        awake_experts=regional_experts[region_id]
  #      print('set of awake experts')
  #      print(awake_experts)
  #      print('awake expert average')
 #       print(awake_experts_avg[region_id])
  #      print('number pulls')
  #      print(number_pulls)
  #      print('\n')
        #find best arm
        lcb_list=[max(awake_experts_avg[region_id][i] - lcb_bound(t, number_pulls[region_id], alpha), 0.0) for i in range(len(awake_experts))] 
#        print(lcb_list)
#        print(min(lcb_list))
        cur_min=min(lcb_list)
        min_index=[i for i, v in enumerate(lcb_list) if v == cur_min]
        if c < cur_min:
            best_arm = -1
            count_rej+=1
            loss_alg += c
        else:
            best_arm = awake_experts[random.choice(min_index)]  #lcb_list.index(min(lcb_list))]  #lcb_list.index(min(lcb_list))
        #update regret
            expert_label = exp_hyp_label(dat[t][0], experts[best_arm],one_d)
            loss_alg += acc_loss(dat[t][1], expert_label)
        # all experts whose acceptance region is a subset of accetange region of best arm. 
            number_pulls[region_id]+=1
            inv_pull = 1.0 / number_pulls[region_id]
            for i in range(len(awake_experts)):
                awake_experts_avg[region_id][i] = acc_loss(dat[t][1], exp_hyp_label(dat[t][0], experts[awake_experts[i]],one_d))  * inv_pull + (1 - inv_pull) * awake_experts_avg[region_id][i]

        save_arms.append(best_arm)
        
                
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1

    return loss_alg_at_return_rounds,count_rej_at_return_rounds,save_arms



def ucbd(c, alpha, experts, dat,return_rounds, one_d,update_matrix,R):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)

    expert_cnt_acc = [0.0] * K
    expert_pulls = [0.0] * K
    expert_hyp_losses = [0.0] * K
    loss_alg = 0
    count_rej=0


    for t in range( T):

        # aggregate lcbs
        expert_lcbs = []
        # update slacks and empirical averages
        for i in range(K):
            if exp_label(dat[t][0], experts[i],one_d,R) != -1:  # expert i accepts
                expert_cnt_acc[i] += 1

        for i in range(K):
            exp_prob_acc = expert_cnt_acc[i] / (t+1)
            exp_prob_rej = 1 - exp_prob_acc
            expert_lcbs.append(max((expert_hyp_losses[i] / max(expert_pulls[i],1)) - lcb_bound(t+1, expert_pulls[i], alpha), 0.0) + max(exp_prob_rej - lcb_bound(t+1, t+1, alpha), 0.0) * c) 
        
        #find best arm
        cur_min=min(expert_lcbs)
        min_index=[i for i, v in enumerate(expert_lcbs) if v == cur_min]
        best_arm = random.choice(min_index)#expert_lcbs.index(min(expert_lcbs)) 
        
        #update algorithm loss
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d,R)
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
        loss_alg += expert_loss
        if expert_label == -1:
            count_rej+=1


        # alld experts whose acceptance region is a subset of accetange region of best arm.                                                                  
        current_update=update_matrix[best_arm,:]
        for i in range(K):
            if current_update[i] == 1.:
                expert_pulls[i] += 1
                expert_hyp_losses[i] += acc_loss(dat[t][1], exp_hyp_label(dat[t][0], experts[i],one_d))

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds

       
# def ucbh(c, alpha, experts, dat,return_rounds,one_d):
#     save_arms=[]
#     loss_alg_at_return_rounds=[]
#     count_rej_at_return_rounds=[]
#     enum_return_rounds=0

#     K = len(experts)
#     T = len(dat)
# #    print '\n\n ucbH'
#     hyp_expert_avg = [0.0]*K #only care of emperical average over hyp_experts.
#     expert_pulls = [0.0]*K #counts the number of times when r>0
#     loss_alg = 0
#     count_rej=0

    
#     for t in range(T):
#         #use dictionary so keep track of which are accepting and rejecting experts at time t

#         acc_exp = 1.0
#         best_acc_exp=-1
#         rej_exp = 1.0
#         best_rej_exp=-1

#         save_expert_labels=[]
#         for i in range(K):
#             #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
#             expert_label = exp_label(dat[t][0], experts[i],one_d)
#             save_expert_labels.append(expert_label)
#             if expert_label!=-1:
#                 if max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < acc_exp:
#                     acc_exp= max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
#                     best_acc_exp=i
#             else:
#                 if max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < rej_exp:
#                     rej_exp=max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
#                     best_rej_exp=i
                    

#         #find best arm
#         if acc_exp < c and best_acc_exp!=-1:
#                 best_arm = best_acc_exp
#                 save_arms.append(best_arm)
#         else:
#             if best_rej_exp!=-1:
#                 best_arm = best_rej_exp
#                 count_rej+=1
#                 save_arms.append(101)
#             else:
#                 best_arm = best_acc_exp
#                 save_arms.append(best_arm)

#         #update regret
#         best_expert_label = save_expert_labels[best_arm]
#         expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
#         #print best_arm, expert_loss,expert_pulls[best_arm]
#         loss_alg += expert_loss
# #        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
#         if best_expert_label != -1:
#             # if pulled a nonrej expert update all acc_experts
#             for jj in range(len(save_expert_labels)):
#                 if save_expert_labels[jj] != -1:
#                             expert_pulls[jj] += 1
#                             inv_pull = 1.0 / expert_pulls[jj]
#                             hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]

#         if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
#             loss_alg_at_return_rounds.append(loss_alg/float(t+1))
#             count_rej_at_return_rounds.append(count_rej/float(t+1))
#             enum_return_rounds+=1
#     return loss_alg_at_return_rounds,count_rej_at_return_rounds,save_arms


############# ############# ############# ############# #############  PLOTTING ############# ############# ############# ############# ############# 
def plotting(c,alpha,K,text_file,ONE_D,TYPE_DATA,R):
#NEED TO IMRPOVE THIS PLOTTING FUNCTN 
    NUM_AVG=4
    T_MAX=10000##700
    avg_regret=[]
    avg_counts=[]
    avg_losses=[]
    avg_edges=[]


#    print experts
    for er in range(NUM_AVG):
            experts,update_matrix,regional_experts= create_experts(K, False, ONE_D,R)
#            print('experts')
#            print(experts)
#            print len(experts)
            x=range(200,T_MAX,200) 

            loss=[]
            count=[]
            expert_loss=[]
            edges=[]
            for p in range(2):
                data=create_data(T_MAX,TYPE_DATA)

                loss_experts,num_reject=loss_of_every_expert(data,experts,c,x,ONE_D,R)    

                loss1,countrej1,savearms1,edges1=ucbn(c,alpha,experts,data,x,ONE_D,update_matrix,R) #returns values of all needed roudns
                loss2,countrej2,savearms2,edges2=ucbb(c,alpha,experts,data,x,ONE_D,update_matrix,R) #returns values of all needed roudns
                loss3,countrej3,savearms3,edges3=ucb(c,alpha,experts,data,x,ONE_D,update_matrix,R) #returns values of all needed roudns
                loss4,countrej4,savearms4,edges4=ftl(c,alpha,experts,data,x,ONE_D,update_matrix,R) #returns values of all needed roudns
#                loss5,countrej5,savearms5,edges5=ftl(c,alpha,experts,data,x,ONE_D,update_matrix,R) #returns values of all needed roudns


                expert_loss.append(loss_experts)
                loss.append([loss1,loss2,loss3,loss4])#,loss5])#,loss6,loss7])
                count.append([countrej1,countrej2,countrej3,countrej4])#,countrej5])#,countrej6,countrej7])
                edges.append([edges1,edges2,edges3,edges4])#,edges5])

            loss=np.mean(np.array(loss),axis=0)
            count=np.mean(np.array(count),axis=0)
            expert_loss=np.mean(np.array(expert_loss),axis=0)
            edges=np.mean(np.array(edges),axis=0)

            best_expert_loss=np.amin(expert_loss,axis=1)
            print best_expert_loss
            regret=loss-np.expand_dims(best_expert_loss,axis=0)
            avg_regret.append(regret)
            avg_losses.append(loss)
            avg_counts.append(count)
            avg_edges.append(edges)
            print "DONE WITH FIRST ROUND"
    std_regret=np.std(np.array(avg_regret),axis=0)
    avg_regret=np.mean(np.array(avg_regret),axis=0)
    std_losses=np.std(np.array(avg_losses),axis=0)
    avg_losses=np.mean(np.array(avg_losses),axis=0)  
    std_counts=np.std(np.array(avg_counts),axis=0)
    avg_counts=np.mean(np.array(avg_counts),axis=0)

    std_edges=np.std(np.array(avg_edges),axis=0)
    avg_edges=np.mean(np.array(avg_edges),axis=0)



    text_file.write('\nPseudo Regret of UCB-type Algorithms for '+str(K)+' arms with c '+str(c)+'_dimension_'+str(int(ONE_D)))
    text_file.write('; regret UCBN:'+str(avg_regret[0])+'; std UCBN:'+str(std_regret[0]))
    text_file.write('; regret UCBB:'+str(avg_regret[1])+'; std UCBB:'+str(std_regret[1]))
    text_file.write('; regret UCB:'+str(avg_regret[2])+'; std UCB:'+str(std_regret[2]))
    text_file.write('; regret FTL:'+str(avg_regret[3])+'; std FTL:'+str(std_regret[3]))
#    text_file.write('; regret FTL:'+str(avg_regret[4])+'; std FTL:'+str(std_regret[4]))

    text_file.write('; losses UCBN:'+str(avg_losses[0])+'; std UCBN:'+str(std_losses[0]))
    text_file.write('; losses UCBB:'+str(avg_losses[1])+'; std UCBB:'+str(std_losses[1]))
    text_file.write('; losses UCB:'+str(avg_losses[2])+'; std UCB:'+str(std_losses[2]))
    text_file.write('; losses FTL:'+str(avg_losses[3])+'; std FLT:'+str(std_losses[3]))
#    text_file.write('; losses FTL:'+str(avg_losses[4])+'; std FTL:'+str(std_losses[4]))

    text_file.write('; counts UCBN:'+str(avg_counts[0])+'; std UCBN:'+str(std_counts[0]))
    text_file.write('; counts UCBB:'+str(avg_counts[1])+'; std UCBB:'+str(std_counts[1]))
    text_file.write('; counts UCB:'+str(avg_counts[2])+'; std UCB:'+str(std_counts[2]))
    text_file.write('; counts FTL:'+str(avg_counts[3])+'; std FTL:'+str(std_counts[3]))
#    text_file.write('; counts FTL:'+str(avg_counts[4])+'; std FTL:'+str(std_counts[4]))

    text_file.write('; edges UCBN:'+str(avg_edges[0])+'; std UCBN:'+str(std_edges[0]))
    text_file.write('; edges UCBB:'+str(avg_edges[1])+'; std UCBB:'+str(std_edges[1]))
#    text_file.write('; edges UCB:'+str(avg_edges[2])+'; std UCB:'+str(std_edges[2]))
#    text_file.write('; edges FTL:'+str(avg_edges[3])+'; std FLT:'+str(std_edges[3]))
#    text_file.write('; edges FTL:'+str(avg_edges[4])+'; std FTL:'+str(std_edges[4]))

    fig, ax = plt.subplots()
    plt.tight_layout()

    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.set_xscale("log", nonposx='clip')
#    ax.set_yscale("log", nonposy='clip')
    ax.errorbar(x, avg_regret[0], yerr=std_regret[0],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_regret[1], yerr=std_regret[1],fmt='-',color='limegreen', label='UCB-B')
    ax.errorbar(x, avg_regret[2], yerr=std_regret[2],fmt='r-',label='UCB')
    ax.errorbar(x, avg_regret[3], yerr=std_regret[3],fmt='b-',label='FLT')
#    ax.errorbar(x, avg_regret[4], yerr=std_regret[4],fmt='m-',label='FLT')
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Regret for c = '+str(c),y=1.02)
    plt.savefig('./regret_c'+str(c)+'_d_'+str(int(ONE_D))+'.eps', format='eps', dpi=1000,bbox_inches='tight')
#    plt.savefig('./regret_c'+str(c)+'_d_'+str(int(ONE_D))+'.png',bbox_inches='tight')

    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.set_xscale("log", nonposx='clip')
 #   ax.set_yscale("log", nonposy='clip')
    ax.errorbar(x, avg_losses[0], yerr=std_losses[0],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_losses[1], yerr=std_losses[1],fmt='-',color='limegreen', label='UCB-B')
    ax.errorbar(x, avg_losses[2], yerr=std_losses[2],fmt='r-', label='UCB')
    ax.errorbar(x, avg_losses[3], yerr=std_losses[3],fmt='b-', label='FTL')
#    ax.errorbar(x, avg_losses[4], yerr=std_losses[4],fmt='m-', label='FTL')
    plt.xlabel('Rounds')
    plt.ylabel(' Losses')
    plt.title('Losses for c = '+str(c),y=1.02)
    plt.savefig('./losses_c'+str(c)+'_d_'+str(int(ONE_D))+'.eps', format='eps', dpi=1000,bbox_inches='tight')
#    plt.savefig('./losses_K'+str(len(experts))+'_c'+str(c)+'_dimension_'+str(int(ONE_D))+'.png',bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.set_xscale("log", nonposx='clip')
    ax.errorbar(x, avg_counts[0], yerr=std_counts[0],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_counts[1], yerr=std_counts[1],fmt='-',color='limegreen', label='UCB-B')
    ax.errorbar(x, avg_counts[2], yerr=std_counts[2],fmt='r-', label='UCB')
    ax.errorbar(x, avg_counts[3], yerr=std_counts[3],fmt='b-', label='FTL')
#    ax.errorbar(x, avg_counts[4], yerr=std_counts[4],fmt='m-', label='FLT')
    plt.xlabel('Rounds')
    plt.ylabel('Fraction Abstained')
    plt.title('Fraction Abstained for c = '+str(c))
#    plt.savefig('./counts_c'+str(c)+'_d_'+str(int(ONE_D))+'.png', dpi=1000,bbox_inches='tight')
    plt.savefig('./counts_c'+str(c)+'_d_'+str(int(ONE_D))+'.eps', format='eps', dpi=1000,bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.set_xscale("log", nonposx='clip')
    ax.errorbar(x, avg_edges[0], yerr=std_edges[0],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_edges[1], yerr=std_edges[1],fmt='-',color='limegreen', label='UCB-B')
    ax.errorbar(x, avg_edges[2], yerr=std_edges[2],fmt='r-', label='UCB')
    ax.errorbar(x, avg_edges[3], yerr=std_edges[3],fmt='b-', label='FTL')
#    ax.errorbar(x, avg_edges[4], yerr=std_edges[4],fmt='m-', label='FTL')
    plt.xlabel('Rounds')
    plt.ylabel('Number of Edges')
    plt.title('Edges for c = '+str(c))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    plt.savefig('./edges_c'+str(c)+'_d_'+str(int(ONE_D))+'.png', dpi=1000,bbox_inches='tight')
    plt.savefig('./edges_c'+str(c)+'_d_'+str(int(ONE_D))+'.eps', format='eps', dpi=1000,bbox_inches='tight')

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"
############# ############# ############# ############# #############  MAIN ############# ############# ############# ############# ############# 
if __name__ == "__main__":

    alpha=3
    val=int(sys.argv[1])-1

#    data=[8,8,8,25,25,25,22,22,22,28,28,28,54,54,54,123,123,123,68,68,68]
#    c_values=[0.1,0.2,0.3]
    data=[8,8,8,8,8,8,22,22,22,22,22,22,54,54,54,54,54,54,68,68,68,68,68,68,25,25,25,25,25,25,14,14,14,14,14,14,28,28,28,28,28,28]
    c_values=[0.1,0.2,0.3,0.001,0.05,0.9]


    K=100
    R=20
    c=c_values[val%6]
    ONE_D=data[val]
    TYPE_DATA=data[val]
    print c
    print ONE_D
    print TYPE_DATA
 
    text_file = open("./Output_" + str(TYPE_DATA) + "d.txt", "a")
    plotting(c,alpha,K,text_file,ONE_D,TYPE_DATA,R) #last plot point is for T=2000                   
    text_file.close()
 
