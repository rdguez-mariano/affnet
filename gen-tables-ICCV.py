from library import *

ds = LoadDatasets()
lda = CPPbridge('./deep-asift/build/libDA.so')
# sift_all = OnlyUniqueMatches(sift_all,KPlist1,KPlist2,SpatialThres=5)
# sift_src_pts = np.float32([ KPlist1[m.queryIdx].pt for m in sift_all ]).ravel()
# sift_dst_pts = np.float32([ KPlist2[m.trainIdx].pt for m in sift_all ]).ravel()
# matchesMask_sift, H_sift = lda.GeometricFilter(sift_src_pts, img1, sift_dst_pts, img2, Filter='ORSA_H')


class MethodScore(object):
    def __init__(self):
        self.score = 0
        self.cInliers = 0
        self.av_dist_pxls = 0
        self.ratioTruePositives = 0
    def AddInfo(self,cmHC,HC,total, AvDist):
        if float(len(cmHC))>0.8*float(len(HC)):
             self.score += 1
             self.cInliers += float(len(cmHC))
             self.ratioTruePositives += float(len(cmHC))/float(len(total))
             self.av_dist_pxls += AvDist
    def GetScore(self, inStr = True):
        if inStr:
            if self.score>0:
                return "C.Imgs: " + str(self.score)+ "; Av. C.In.: " + str(self.cInliers/self.score) + "; Av. Error: " + str(self.av_dist_pxls/self.score) +"; TruePos: " +str(self.ratioTruePositives/self.score)
            else:
                return 'No success in this dataset'
        else:
            if self.score>0:
                return self.score, self.cInliers/self.score, self.av_dist_pxls/self.score, self.ratioTruePositives/self.score
            else:
                return 0, -1, -1


do_listnames = False
listnames = ('adam', 'graf', 'pkk', 'mag')


verb = False
for i in range(3):
    AN = MethodScore()

    for p in ds[i].datapairs:
            if (not do_listnames) or (p.pair_name in listnames):
                if verb:                
                    print(p.pair_name)
                total, good_HC, kplistq, kplistt, H, ET_KP, ET_M = AffNet(p.query,p.target, Visual = False, cvimg1=p.queryimg, cvimg2=p.targetimg)
                cmHC, AvDist = CorrectMatches(good_HC,kplistq, kplistt, p.Tmatrix )
                cmT,_ = CorrectMatches(total,kplistq, kplistt, p.Tmatrix )    
                AN.AddInfo(cmHC, good_HC, total, AvDist)
                if verb:
                    print("----> Affnet : cmT = %d, cmHC = %d(%3.2f), HC = %d, Total = %d, ET_KP = %3.3f, ET_M = %3.3f" %(len(cmT), len(cmHC), AvDist, len(good_HC), len(total), ET_KP,ET_M))
    print('Resume on',ds[i].name)
    print('---->     HessAffnet : ',AN.GetScore())


