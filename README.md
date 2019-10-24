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
dev_acc: 0.8502613306045532   
test_acc: 0.8411999940872192   
dev_loss: 0.5807797582581394   
test_loss: 0.5418186249828338