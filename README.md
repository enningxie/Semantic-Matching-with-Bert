# Semantic-Matching-with-Bert
Semantic matching with bert.


---

##### 实验记录
  
model_name: test_albert_tiny_01.h5  
weight: albert_tiny_250k  
data_type: LCQMC  
mode: part  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.8494660258293152   
test_acc: 0.8481600284576416   
dev_loss: 0.4105272938372934   
test_loss: 0.37515994957923887

---

model_name: test_albert_tiny_02.h5  
weight: albert_tiny_250k  
data_type: LCQMC  
mode: full  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.8502613306045532   
test_acc: 0.8411999940872192   
dev_loss: 0.5807797582581394   
test_loss: 0.5418186249828338

---

model_name: test_albert_tiny_03.h5  
weight: albert_tiny_250k  
data_type: BQ  
mode: part  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.8130999803543091   
test_acc: 0.7991999983787537   
dev_loss: 0.5380937144756317   
test_loss: 0.5708155479431153

---

model_name: test_albert_tiny_04.h5  
weight: albert_tiny_250k  
data_type: BQ  
mode: full  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.8141000270843506   
test_acc: 0.7958999872207642   
dev_loss: 0.48957341678142546   
test_loss: 0.5119842576980591

---

model_name: test_albert_tiny_05.h5  
weight: albert_tiny_250k  
data_type: sent_pair  
mode: part  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.831932783126831   
test_acc: 0.8264889121055603   
dev_loss: 0.44651593383785615   
test_loss: 0.4599930316368739

---

model_name: test_albert_tiny_06.h5  
weight: albert_tiny_250k  
data_type: sent_pair  
mode: full  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.8282629251480103   
test_acc: 0.8172444701194763   
dev_loss: 0.47713068260143265   
test_loss: 0.4963957167810864

---

model_name: test_albert_tiny_07.h5  
weight: albert_tiny_250k  
data_type: LCQMC  
mode: full  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.8282629251480103  
test_acc: 0.8172444701194763  
dev_loss: 0.47713068260143265  
test_loss: 0.4963957167810864  
desc: 质疑test_albert_tiny_02的表现，复现

---

model_name: test_albert_tiny_08.h5  
weight: albert_tiny_250k  
data_type: LCQMC  
mode: full  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.8282629251480103  
test_acc: 0.8172444701194763  
dev_loss: 0.47713068260143265  
test_loss: 0.4963957167810864

---

model_name: test_albert_tiny_09.h5  
weight: albert_tiny_250k  
data_type: LCQMC  
mode: full  
max_len: 32  
batch_size: 64  
learning_rate: 1e-4  
dev_acc: 0.8282629251480103  
test_acc: 0.8172444701194763  
dev_loss: 0.47713068260143265  
test_loss: 0.4963957167810864