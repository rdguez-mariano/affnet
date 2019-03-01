from library import *

ds = LoadDatasets()
lda = CPPbridge('./deep-asift/build/libDA.so')
# sift_all = OnlyUniqueMatches(sift_all,KPlist1,KPlist2,SpatialThres=5)
# sift_src_pts = np.float32([ KPlist1[m.queryIdx].pt for m in sift_all ]).ravel()
# sift_dst_pts = np.float32([ KPlist2[m.trainIdx].pt for m in sift_all ]).ravel()
# matchesMask_sift, H_sift = lda.GeometricFilter(sift_src_pts, img1, sift_dst_pts, img2, Filter='ORSA_H')

for p in ds[0].datapairs:
    print(p.pair_name)
    total, good_HC, kplistq, kplistt, H, ET_KP, ET_M = AffNet(p.query,p.target, Visual = False, cvimg1=p.queryimg, cvimg2=p.targetimg)
    cmHC = CorrectMatches(good_HC,kplistq, kplistt, p.Tmatrix )
    cmT = CorrectMatches(total,kplistq, kplistt, p.Tmatrix )    
    print("----> Affnet : cmT = %d, cmHC = %d, HC = %d, Total = %d, ET_KP = %3.3f, ET_M = %3.3f" %(len(cmT), len(cmHC), len(good_HC), len(total), ET_KP,ET_M))