
padd=1

f1=open("./dat.txt","r")

conv_res=f1.readlines()
layer=0
for s in range(len(conv_res)):
    if "conv input" in conv_res[s]:
        layer+=1
    # if layer==29:
        x=conv_res[s+1].split(" ")[:-1]
        w=conv_res[s+3].split(" ")[:-1]
        y_=conv_res[s+5].split(" ")[:-1]
        c=int(conv_res[s].split(" ")[2])
        w_in=int(conv_res[s].split(" ")[3])
        d=int(conv_res[s+4].split(" ")[2])
        y_=[int(i) for i in y_]
        w=[int(i) for i in w]
        x=[int(i) for i in x]

        # w_in*w_in conv 3*3 (pad=1) = w_in*w_in
        import math
        def get_padd(x):
            return 2**int(math.log2(x-1)+1)

        padd_w=(w_in+2*padd)

        X=[] # padded X
        W=[0 for i in range(get_padd(padd_w*3)*get_padd(c)*get_padd(d))] # converted w, size: (w_in+2)*3

        for co in range(d):
            for ci in range(c):
                for i in range(3):
                    for j in range(3):
                        W[(co*get_padd(c)+ci)*get_padd(padd_w*3)+padd_w*i+j]=int(w[co*c*3*3+ci*3*3+3*i+j])
        

        len_w=padd_w*3
        for ci in range(get_padd(c)):
            xx=[]
            for i in range(padd_w*padd_w):
                xi=i//padd_w
                yi=i%padd_w
                if ci>=c:
                    xx.append(0)
                    continue
                if xi==0 or yi==0 or xi==padd_w-1 or yi==padd_w-1: # in padded area
                    xx.append(0)
                else:
                    xx.append(int(x[ci*w_in*w_in+(w_in-1-(xi-1))*w_in+w_in-1-(yi-1)]))
                
            len_x=len(xx)

            for i in range(get_padd(len_x)-len_x):
                xx.append(0)
            X=X+xx


        # pad x=> X: degree (w_in+2)*(w_in+2)


        Y=[0 for i in range(get_padd(len_x+len_w)*d)]


        # Y: degree (w_in+2)*(w_in+5)
        PADD_Y=get_padd(len_x+len_w)
        PADD_X=get_padd(len_x)
        PADD_W=get_padd(len_w)
        PADD_C=get_padd(c)
        y=[]
        for co in range(d):
            for i in range(w_in**2):
                y.append(y_[co*w_in*w_in+i])
            for i in range(PADD_Y-w_in*w_in):
                y.append(0)
        # conv:
        import time
        t1=time.time()
        # print(get_padd(d)*get_padd(c)*len_x*len_w)
        for co in range(get_padd(d)):
            for ci in range(get_padd(c)):
                for i in range(len_x):
                    for j in range(len_w):
                        Y[co*PADD_Y+i+j]+=X[ci*PADD_X+i]*W[(co*PADD_C+ci)*PADD_W+j]
        t2=time.time()

        # permute Y to obtain y
        visit=[0 for i in range(PADD_Y)]
        for co in range(d):
            for i in range(w_in*w_in):
                xi=i//w_in
                yi=i%w_in
                visit[(padd_w-1-xi)*padd_w+padd_w-1-yi]=1  # find unused positions, these are elements in P
                assert(y[co*PADD_Y+i]==Y[co*PADD_Y+(padd_w-1-xi)*padd_w+padd_w-1-yi])
        p_pos=[]
        P=[0 for i in range(d*PADD_Y)]
        
        for i in range(PADD_Y):
                if i<len_x+len_w: # not zero
                    if visit[i]==0:
                        p_pos.append(i)
                else: # padding region
                    break
        for co in range(d):
            for i in range(len(p_pos)):
                P[co*PADD_Y+i]=Y[co*PADD_Y+p_pos[i]]
        for co in range(d):
            for i in range(len(p_pos)):
                assert(P[co*PADD_Y+i]==Y[co*PADD_Y+p_pos[i]])
        assert(len(Y)==len(y))
        assert(len(Y)==len(P))
        out=open(f"dat/conv_layer_{layer}.txt","w")
        print("plain x:",(c,w_in**2),x,file=out)
        print("rot pad x: ",(get_padd(c),PADD_X),X,file=out)
        print("weight W: ",(get_padd(d),get_padd(c),PADD_W),W,file=out)
        print("conv direct Y: ",(get_padd(d),PADD_Y),Y,file=out)
        print("rot y: ","useful shape: ",(get_padd(d),w_in**2),"padded shape: ",(get_padd(d),PADD_Y),y,file=out)
        print("P: ","useful shape: ",(get_padd(d),len(p_pos)),"padded shape: ",(get_padd(d),PADD_Y),P,file=out)
        out.close()
    elif "relu input" in conv_res[s]:
        layer+=1
        out=open(f"dat/relu_layer_{layer}.txt","w")
        print(conv_res[s],end="",file=out)
        print(conv_res[s+1],end="",file=out)
        print(conv_res[s+2],end="",file=out)
        print(conv_res[s+3],end="",file=out)
        print(conv_res[s+4],end="",file=out)
        print(conv_res[s+5],end="",file=out)
        print(conv_res[s+6],end="",file=out)
        print(conv_res[s+7],end="",file=out)
        out.close()
    elif "max pool in:" in conv_res[s]:
        layer+=1
        out=open(f"dat/maxpool_layer_{layer}.txt","w")
        print(conv_res[s],end="",file=out)
        print(conv_res[s+1],end="",file=out)
        print(conv_res[s+2],end="",file=out)
        print(conv_res[s+3],end="",file=out)
        out.close()
    elif "matrix mult weight" in conv_res[s]:
        layer+=1
        out=open(f"dat/matrix_layer_{layer}.txt","w")
        print(conv_res[s],end="",file=out)
        print(conv_res[s+1],end="",file=out)
        print(conv_res[s+2],end="",file=out)
        out.close()
    print("finish: ",s,len(conv_res))
    # X shape: [4,2048]: 8192
    # W shape: [4,64,128]: 32768
    # Y shape: [64,2048]: 131072


