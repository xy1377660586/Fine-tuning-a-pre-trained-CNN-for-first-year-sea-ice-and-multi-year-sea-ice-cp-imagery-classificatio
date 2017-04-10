#including the loss plot

niter = 1000
  
# losses will also be stored in the log  
train_loss = np.zeros(niter)  
scratch_train_loss = np.zeros(niter)  
test_label= np.zeros(niter)
  
caffe.set_device(0)  
caffe.set_mode_gpu()  
# We create a solver that fine-tunes from a previously trained network.  
solver = caffe.SGDSolver('myfinetunning/data/solver.prototxt')  
solver.net.copy_from('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')  
# For reference, we also create a solver that does no finetuning.  
scratch_solver = caffe.SGDSolver('myfinetunning/data/solver.prototxt') 
 
  
# We run the solver for niter times, and record the training loss.  
for it in range(niter):  
    solver.step(1)  # SGD by Caffe  
    scratch_solver.step(1)  
    # store the train loss  
    train_loss[it] = solver.net.blobs['loss'].data  
    scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data 
    test_label[it]=solver.net.blobs['label'].data 
    if it % 10 == 0:  
        print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])  
print 'done'  
sio.savemat('/home/lein/caffe/myfinetunning/data/train_loss',{'data':train_loss})
sio.savemat('/home/lein/caffe/myfinetunning/data/loss_scrach',{'data':scratch_train_loss})


sio.savemat('/home/lein/caffe/myfinetunning/data/pred_label',{'data':test_label})



