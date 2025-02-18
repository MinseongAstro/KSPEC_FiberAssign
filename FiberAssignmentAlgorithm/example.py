"""
fiber configuration, all length units are [mm]
"""
alpha_length=5.2 #fiber positioner alpha arm length
beta_length=11.6 #fiber positioner beta arm length

fid_list=np.array([0,6,11,33,36,41,51,73,77,84,91,95,117,127,132,135,157,161,168]) #fudcial & guiding camera index
#73 -> 72  & 95 -> 96 ? fiducial changed ?
x,y=[],[]
xx=[]

d=16.8
n=8

adj_fib=[]


for i in range(2*n-1):
    if i<8 : 
        rownum=8+i
    else : 
        rownum=22-i
    for j in range(rownum):
            x.append((j-(rownum-1)/2)*d)
            y.append((n-1-i)*d*np.sqrt(3)/2)
x=np.delete(x,fid_list)
y=np.delete(y,fid_list)

for i in range(len(x)):
    r=pow(x-x[i],2)+pow(y-y[i],2)
    idx=np.where(r<4*16.8**2+0.01)
    adj_fib.append([item for item, count in Counter(idx[0]).items() if (count > 0) & (item != i)])
            

fiber_pos=pd.DataFrame({"x":x,"y":y,"fiber_num":np.arange(1,len(x)+1),"adj_fib_idx":adj_fib})

def avoid_fiducial(obj_x,obj_y):
    idx0=np.arange(len(obj_x)); sel=np.zeros(len(obj_x))
    d=16.8; n=8
    x,y=[],[]; fid_list=[0,6,11,33,36,41,51,73,77,84,91,95,117,127,132,135,157,161,168]
    #73 -> 72  & 95 -> 96 ? fiducial changed ?
    for i in range(2*n-1):
        if i<8 : 
            rownum=8+i
        else : 
            rownum=22-i
        for j in range(rownum):
            x.append((j-(rownum-1)/2)*d)
            y.append((n-1-i)*d*np.sqrt(3)/2)
    fid_x=(np.array(x))[fid_list]
    fid_y=(np.array(y))[fid_list]
    fid_col=fid_x*0.+3.5; fid_col[9]=3.5
    for i in np.arange(len(fid_x)):
        sel[idx0[np.where((obj_x-fid_x[i])**2+(obj_y-fid_y[i])**2 <= fid_col[i]**2)]]=1.
    return sel

### example targets/tiles
main0_ra, main0_dec, main0_sz0, main0_mk = np.loadtxt('./targetlist.txt',usecols=[1,2,3,9]).T
selm=np.where((main0_mk<=13.75)&(main0_sz0==-9.))
main_ra=main0_ra[selm]; main_dec=main0_dec[selm]; main_rank=main0_mk[selm] #8.752-13.75

qso0 = readsav('./cat_qso_selc.sav')
selq=np.where(qso0['qso_sz']==99.)
qso_ra=qso0['qso_ra'][selq]; qso_dec=qso0['qso_dec'][selq]; qso_rank=qso0['qso_mr'][selq]*0+150

tile0_ra, tile0_dec = np.loadtxt('./tile_result_max_NEW32.txt',usecols=[0,1]).T
###

fibn=150
alpha_length=5.2 
beta_length=11.6 
col_cond=3.5
thread_num=8
max_loop=1e7

obj_ra  =main_ra
obj_dec =main_dec
obj_rank=main_rank
numx=len(obj_ra)

result_id=np.arange(fibn)+1
print('start =', (datetime.now()).strftime("%H:%M:%S"))

for nn in np.arange(1):
    tile_ra=tile0_ra[nn]; tile_dec=tile0_dec[nn]

    result_x    =np.zeros(fibn)+999.
    result_y    =np.zeros(fibn)+999.
    result_rank =np.zeros(fibn)-1.
    
    dely=abs(obj_dec-tile_dec)
    delx=abs(obj_ra-tile_ra)
    delx1=abs(obj_ra-tile_ra-360); delx2=abs(obj_ra-tile_ra+360)
    if abs(tile_dec)>60:
        idx1=np.where(dely<1.5)
    if (abs(tile_dec)<=60)&((tile_ra>=10)&(tile_ra<=350)):
        idx1=np.where((dely<1.5)&(delx<3.))
    if (abs(tile_dec)<=60)&(tile_ra<10):
        delx[np.where(obj_ra-tile_ra>= 180)]=delx1[np.where(obj_ra-tile_ra>= 180)]
        idx1=np.where((dely<1.5)&(delx<3.))
    if (abs(tile_dec)<=60)&(tile_ra>350):
        delx[np.where(obj_ra-tile_ra<=-180)]=delx2[np.where(obj_ra-tile_ra<=-180)]
        idx1=np.where((dely<1.5)&(delx<3.))
    if len(obj_dec[idx1]) > 0 :
        x,y=TPV_WCS(obj_ra[idx1],obj_dec[idx1],tile_ra,tile_dec)
        #sel=avoid_fiducial(x,y)
        idx2=np.where((x*x+y*y<pow(16.8*8,2)))#&(sel==0))
        if len(obj_dec[idx1][idx2]) > 0 :
            rankx=obj_rank[idx1][idx2]  
            objx=pd.DataFrame({"x":x[idx2],"y":y[idx2],"rank":rankx})
            objx=objx.append(objx.iloc[0],ignore_index=True)
            objx.iloc[0]['x']=150
            objx.iloc[0]['y']=150
            objx.iloc[0]['rank']=150
            
            t1 = time.time()
            obj_3_assigned,fiber_pos_3,obj_3=proc1(objx,fiber_pos)
            orig_assigned=obj_3_assigned.copy(deep=True)
            group=proc2(obj_3,obj_3_assigned,fiber_pos_3,alpha_length,beta_length,col_cond)         
            obj_3_assigned=proc3(orig_assigned,fiber_pos_3,group,obj_3,obj_3_assigned,thread_num,alpha_length,beta_length,max_loop,col_cond)
            t2 = time.time()
    for i in np.arange(0,fibn):
        idx=np.where(obj_3_assigned['fiber_num'] == result_id[i])
        if len(result_id[idx]) == 1:
            result_x[i]    =(np.array(obj_3_assigned["x"]))[idx]
            result_y[i]    =(np.array(obj_3_assigned["y"]))[idx]
            result_rank[i]=(np.array(obj_3_assigned["rank"]))[idx]

    
    #print(obj_3_assigned)
    #print(proc2(obj_3,obj_3_assigned,fiber_pos_3,alpha_length,beta_length,col_cond))
    dt = (t2-t1)
    print(nn+1,' done =',(datetime.now()).strftime("%H:%M:%S"),len(result_rank[np.where((result_rank>0)&(result_rank<20))]),len(objx)-1,np.round(dt,3))
    np.savetxt('./target_test/'+'%4.4i'%(nn+1)+'_result.txt',[len(result_rank[np.where((result_rank>0)&(result_rank<20))]),len(objx)-1,dt])
