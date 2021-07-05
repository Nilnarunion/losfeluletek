random_direction1 = get_random_weights(copy_of_the_weights)
random_direction2 = get_random_weights(copy_of_the_weights)

for d1,d2,w in zip(random_direction1,
		random_direction2,
		copy_of_the_weights):
    
    if w.dim() == 1:
        d1.data = torch.zeros_like(w)
        d2.data = torch.zeros_like(w)
        
    elif w.shape[0] == 10:
        d11,_ = tf.qr(d1.cpu().numpy())
        d11   = d11.eval()
        d22,_ = tf.qr(np.transpose(d2.cpu().numpy(),
        			(2,3,1,0)))
        d22   = np.transpose(d22.eval(),(3,2,0,1))
        

        d1.data   = torch.from_numpy(d11).cuda()
        d2.data   = torch.from_numpy(d22).cuda()     
        
        w_norm  = w.view((w.shape[0],-1)).norm(dim=(1),	
        			keepdim=True)[:,:,None,None]
        d_norm1 = d1.view((d1.shape[0],-1)).norm(dim=(1),
        			keepdim=True)[:,:,None,None]
        d_norm2 = d2.view((d2.shape[0],-1)).norm(dim=(1),
        			keepdim=True)[:,:,None,None]

        d1.data = d1.cuda() * (w_norm/(d_norm1.cuda()+1e-10))
        d2.data = d2.cuda() * (w_norm/(d_norm2.cuda()+1e-10))
        
    else:
        d11,_ = tf.qr(d1.cpu().numpy())
        d11   = d11.eval()
        d22,_ = tf.qr(np.transpose(d2.cpu().numpy(),(2,3,0,1)))
        d22   = np.transpose(d22.eval(),(2,3,0,1))
        print(d11.shape,d22.shape)

        d1.data   = torch.from_numpy(d11).cuda()
        d2.data   = torch.from_numpy(d22).cuda()     
        
        w_norm  = w.view((w.shape[0],-1))  .norm(dim=(1),
        			keepdim=True)[:,:,None,None]
        d_norm1 = d1.view((d1.shape[0],-1)).norm(dim=(1),
        			keepdim=True)[:,:,None,None]
        d_norm2 = d2.view((d2.shape[0],-1)).norm(dim=(1),
        			keepdim=True)[:,:,None,None]

        d1.data = d1.cuda() * (w_norm/(d_norm1.cuda()+1e-10))
        d2.data = d2.cuda() * (w_norm/(d_norm2.cuda()+1e-10))