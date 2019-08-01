import torch
import torch.nn as nn
import numpy as np
import sys
import glob, os
import time

from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A
from Utils import line_prepender
from architectures import AffNetFast
from HardNet import HardNet

import ctypes
import cv2
sys.path.append("./deep-asift/")

class GTPairs(object):
    def __init__(self,name, queryimg, targetimg, query,target,T):
        self.pair_name = name
        self.queryimg = queryimg
        self.targetimg = targetimg
        self.query = query
        self.target = target
        self.Tmatrix = T 

class DatasetClass(object):
    def __init__(self,datasets_path, Ttype = 'Homography', name=''):
        self.name = name
        self.Ttype = Ttype
        self.path = datasets_path
        self.datapairs = []

def LoadDatasets():
    datasets = []

    # EVD        
    ds_path = 'deep-asift/acc-test/TestDatasets/EVD'
    f = DatasetClass(ds_path, name = 'EVD')
    for file in glob.glob(f.path+"/1/*"):
        f.datapairs.append( GTPairs(
            os.path.basename(file)[:-4],
            cv2.cvtColor( cv2.imread(os.path.join(ds_path,'1',os.path.basename(file)[:-4])+'.png') ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor( cv2.imread(os.path.join(ds_path,'2',os.path.basename(file)[:-4])+'.png') ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            os.path.join(ds_path,'1',os.path.basename(file)[:-4])+'.png',
            os.path.join(ds_path,'2',os.path.basename(file)[:-4])+'.png',
            np.loadtxt(os.path.join(ds_path,'h',os.path.basename(file)[:-4]+'.txt'))
        ) )
    datasets.append( f )

    # OxAff        
    ds_path = 'deep-asift/acc-test/TestDatasets/OxAff'
    f = DatasetClass(ds_path, name = 'OxAff')
    for tdir in glob.glob(f.path+"/*"):
        ext = glob.glob(tdir+"/img1.*")[0][-4:]
        for i in range(2,7):
            f.datapairs.append( GTPairs(
                os.path.basename(tdir)+'_1_to_'+str(i),
                cv2.cvtColor( cv2.imread(tdir+'/img1'+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
                cv2.cvtColor( cv2.imread(tdir+'/img'+str(i)+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
                tdir+'/img1'+ext,
                tdir+'/img'+str(i)+ext,
                np.loadtxt(tdir+'/H1to'+str(i)+'p')
            ) )
    datasets.append( f )

    # SymB        
    ds_path = 'deep-asift/acc-test/TestDatasets/SymB'
    f = DatasetClass(ds_path, name = 'SymB')
    for tdir in glob.glob(f.path+"/*"):
        ext = glob.glob(tdir+"/01.*")[0][-4:]        
        f.datapairs.append( GTPairs(
            os.path.basename(tdir),
            cv2.cvtColor( cv2.imread(tdir+'/01'+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor( cv2.imread(tdir+'/02'+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            tdir+'/01'+ext,
            tdir+'/02'+ext,
            np.loadtxt(tdir+'/H1to2')
        ) )
    datasets.append( f )

        # GDB        
    ds_path = 'deep-asift/acc-test/TestDatasets/GDB'
    f = DatasetClass(ds_path, name = 'GDB', Ttype = 'None')
    for file in glob.glob(f.path+"/*"):
        f.datapairs.append( GTPairs(
            os.path.basename(file)[:-4],
            cv2.cvtColor( cv2.imread(os.path.join(ds_path+'/'+os.path.basename(file)[:-5]+'1'+os.path.basename(file)[-4:])) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor( cv2.imread(os.path.join(ds_path+'/'+os.path.basename(file)[:-5]+'2'+os.path.basename(file)[-4:])) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            os.path.join(ds_path+'/'+os.path.basename(file)[:-5]+'1'+os.path.basename(file)[-4:]),
            os.path.join(ds_path+'/'+os.path.basename(file)[:-5]+'2'+os.path.basename(file)[-4:]),
            None
        ) )
    datasets.append( f )

    return datasets

    
def CorrectMatches(matches,kplistq, kplistt, H, thres = 24):
    goodM = []
    AvDist = 0
    for m in matches:
        x = kplistq[m.queryIdx].pt + tuple([1])
        x = np.array(x).reshape(3,1)
        Hx = np.matmul(H, x)
        Hx = Hx/Hx[2]
        
        y =kplistt[m.trainIdx].pt
        thisdist = cv2.norm(Hx[0:2],y)
        if  thisdist <= thres:
            goodM.append(m)
            AvDist += thisdist
    if len(goodM)>0:                
        AvDist = AvDist/float(len(goodM))
    else:
        AvDist = -1    
    return goodM, AvDist


def OnlyUniqueMatches(goodM, KPlistQ, KPlistT, SpatialThres=4):
    ''' Filter out non unique matches with less similarity score
    '''
    uniqueM = []
    doubleM = np.zeros(len(goodM),dtype=np.bool)
    for i in range(0,len(goodM)):
        if doubleM[i]:
            continue
        bestsim = goodM[i].distance
        bestidx = i
        for j in range(i+1,len(goodM)):
            if  ( cv2.norm(KPlistQ[goodM[i].queryIdx].pt, KPlistQ[goodM[j].queryIdx].pt) < SpatialThres \
            and   cv2.norm(KPlistT[goodM[i].trainIdx].pt, KPlistT[goodM[j].trainIdx].pt) < SpatialThres ):
                doubleM[j] = True
                if bestsim<goodM[j].distance:
                    bestidx = j
                    bestsim = goodM[j].distance
        uniqueM.append(goodM[bestidx])
    return uniqueM


class CPPbridge(object):
    def __init__(self,libDApath):
        self.libDA = ctypes.cdll.LoadLibrary(libDApath)
        self.MatcherPtr = 0
        self.last_i1_list = []
        self.last_i2_list = []

        self.libDA.GeometricFilter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libDA.GeometricFilter.restype = None

        self.libDA.GeometricFilterFromNodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libDA.GeometricFilterFromNodes.restype = None
        self.libDA.ArrayOfFilteredMatches.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        self.libDA.GeometricFilterFromNodes.restype = None
        self.libDA.NumberOfFilteredMatches.argtypes = [ctypes.c_void_p]
        self.libDA.NumberOfFilteredMatches.restype = ctypes.c_int

        self.libDA.newMatcher.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.libDA.newMatcher.restype = ctypes.c_void_p
        self.libDA.KnnMatcher.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libDA.KnnMatcher.restype = None

        self.libDA.GetData_from_QueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.GetData_from_QueryNode.restype = None
        self.libDA.GetQueryNodeLength.argtypes = [ctypes.c_void_p]
        self.libDA.GetQueryNodeLength.restype = ctypes.c_int

        self.libDA.LastQueryNode.argtypes = [ctypes.c_void_p]
        self.libDA.LastQueryNode.restype = ctypes.c_void_p
        self.libDA.FirstQueryNode.argtypes = [ctypes.c_void_p]
        self.libDA.FirstQueryNode.restype = ctypes.c_void_p
        self.libDA.NextQueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.NextQueryNode.restype = ctypes.c_void_p
        self.libDA.PrevQueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.PrevQueryNode.restype = ctypes.c_void_p

        self.libDA.FastMatCombi.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.FastMatCombi.restype = None

    def GeometricFilter(self, scr_pts, im1, dts_pts, im2, Filter = 'USAC_H', precision = 10, verb=False):
        filtercode=0
        if Filter=='ORSA_F':
            filtercode=1
        elif Filter=='USAC_H':
            filtercode=2
        elif Filter=='USAC_F':
            filtercode=3
        N = int(len(scr_pts)/2)
        scr_pts = scr_pts.astype(ctypes.c_float)
        dts_pts = dts_pts.astype(ctypes.c_float)
        MatchMask = np.zeros(N, dtype = ctypes.c_bool)
        T = np.zeros(9, dtype = ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        boolp = ctypes.POINTER(ctypes.c_bool)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        self.libDA.GeometricFilter(scr_pts.ctypes.data_as(floatp), dts_pts.ctypes.data_as(floatp),
                                    MatchMask.ctypes.data_as(boolp), T.ctypes.data_as(floatp),
                                    N, w1, h1, w2, h2, filtercode, ctypes.c_float(precision), verb)
        return MatchMask.astype(np.bool), T.astype(np.float).reshape(3,3)
    def FirstLast_QueryNodes(self):
        return self.libDA.FirstQueryNode(self.MatcherPtr), self.libDA.LastQueryNode(self.MatcherPtr)

    def NextQueryNode(self, qn):
        return self.libDA.NextQueryNode(self.MatcherPtr, qn)

    def PrevQueryNode(self, qn):
        return self.libDA.PrevQueryNode(self.MatcherPtr, qn)

    def KnnMatch(self,QKPlist,Qdesc, TKPlist, Tdesc, FastCode):
        Nq = ctypes.c_int(np.shape(Qdesc)[0])
        Nt = ctypes.c_int(np.shape(Tdesc)[0])
        Qkps = np.array([x for kp in QKPlist for x in kp.pt],dtype=ctypes.c_float)
        Tkps = np.array([x for kp in TKPlist for x in kp.pt],dtype=ctypes.c_float)        
        floatp = ctypes.POINTER(ctypes.c_float)
        Qdesc = Qdesc.ravel().astype(ctypes.c_float)
        Tdesc = Tdesc.ravel().astype(ctypes.c_float)
        QdescPtr = Qdesc.ctypes.data_as(floatp)
        TdescPtr = Tdesc.ctypes.data_as(floatp)
        QkpsPtr = Qkps.ctypes.data_as(floatp)
        TkpsPtr = Tkps.ctypes.data_as(floatp)
        
        self.libDA.KnnMatcher(self.MatcherPtr,QkpsPtr, QdescPtr, Nq, TkpsPtr, TdescPtr, Nt, ctypes.c_int(FastCode))

    def CreateMatcher(self,desc_dim, k=1, sim_thres=0.7):
        self.MatcherPtr = self.libDA.newMatcher(k,desc_dim,sim_thres)


USE_CUDA = False
USE_GPU = False

lda = CPPbridge('./deep-asift/build/libDA.so')

### Initialization
AffNetPix = AffNetFast(PS = 32)
weightd_fname = 'pretrained/AffNet.pth'

if USE_GPU:
    checkpoint = torch.load(weightd_fname)
else:
    checkpoint = torch.load(weightd_fname, map_location='cpu')
AffNetPix.load_state_dict(checkpoint['state_dict'])
AffNetPix.eval()
    
detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 3000,
                                        border = 5, num_Baum_iters = 1, 
                                        AffNet = AffNetPix)
descriptor = HardNet()
model_weights = 'HardNet++.pth'
if USE_GPU:
    hncheckpoint = torch.load(model_weights)
else:
    hncheckpoint = torch.load(model_weights, map_location='cpu')
descriptor.load_state_dict(hncheckpoint['state_dict'])
descriptor.eval()
if USE_CUDA:
    detector = detector.cuda()
    descriptor = descriptor.cuda()


def load_grayscale_var(fname):
    img = Image.open(fname).convert('RGB')
    img = np.mean(np.array(img), axis = 2)
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)))
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    if USE_CUDA:
        var_image_reshape = var_image_reshape.cuda()
    return var_image_reshape


## Detection and description
def get_geometry_and_descriptors(img, det, desc):
    with torch.no_grad():
        LAFs, resp = det(img, do_ori = True)
        patches = detector.extract_patches_from_pyr(LAFs, PS = 32)
        descriptors = descriptor(patches)
    return LAFs, descriptors

from Losses import distance_matrix_vector
from LAF import visualize_LAFs, LAF2pts, convertLAFs_to_A23format
import seaborn as sns

def getCVMatches(LAFs):
    work_LAFs = convertLAFs_to_A23format(LAFs)
    KPlist = []
    for i in range(len(work_LAFs)):
        center = tuple(LAF2pts(work_LAFs[i,:,:], n_pts=0)[0])
        KPlist.append(
            cv2.KeyPoint(x = float(center[0]), y = float(center[1]),
                    _size = 2.0*1.6*pow(2.0, 1/3)*pow(2.0,1),
                    _angle = 0, _response = 0.9, _octave = 1, _class_id = 0)
            )
    return KPlist


def AffNet(input_img_fname1,input_img_fname2, Visual = True, cvimg1 = None, cvimg2 = None):
    output_img_fname = 'deep-asift/temp/kpi_match.png'
    img1 = load_grayscale_var(input_img_fname1)
    img2 = load_grayscale_var(input_img_fname2)

    start_time = time.time()
    LAFs1, descriptors1 = get_geometry_and_descriptors(img1, detector, descriptor)
    LAFs2, descriptors2 = get_geometry_and_descriptors(img2, detector, descriptor)
    ET_KP = time.time() - start_time

    #Bruteforce matching with SNN threshold    
    start_time = time.time()
    SNN_threshold = 0.8

    dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
    min_dist, idxs_in_2 = torch.min(dist_matrix,1)
    dist_matrix[:,idxs_in_2] = 100000;# mask out nearest neighbour to find second nearest
    min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix,1)
    mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold

    tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False)[mask]
    tent_matches_in_2 = idxs_in_2[mask]

    tent_matches_in_1 = tent_matches_in_1.data.cpu().long()
    tent_matches_in_2 = tent_matches_in_2.data.cpu().long()
    ET_M = time.time() - start_time

    KPlist1 = getCVMatches(LAFs1[tent_matches_in_1,:,:].cpu().numpy().squeeze())
    KPlist2 = getCVMatches(LAFs2[tent_matches_in_2,:,:].cpu().numpy().squeeze())
    sift_all = OnlyUniqueMatches( [cv2.DMatch(i, i, 1.0) for i in range(0,len(KPlist1))], KPlist1, KPlist2 )


    sift_consensus = []
    H_sift = [[0, 0, 0], [0, 0, 0],[0, 0, 0]]
    sift_src_pts = np.float32([ KPlist1[m.queryIdx].pt for m in sift_all ]).ravel()
    sift_dst_pts = np.float32([ KPlist2[m.trainIdx].pt for m in sift_all ]).ravel()
    matchesMask_sift, H_sift = lda.GeometricFilter(sift_src_pts, cvimg1, sift_dst_pts, cvimg2, Filter='USAC_H',verb=False)

    for i in range(0,len(matchesMask_sift)):
        if matchesMask_sift[i]==True:
            sift_consensus.append(sift_all[i])
 

    if Visual:
        img4 = cv2.drawMatches(cvimg1,KPlist1,cvimg2,KPlist2,sift_all, None,flags=2)
        cv2.imwrite('./deep-asift/temp/Affnet_matches.png',img4)
        img4 = cv2.drawMatches(cvimg1,KPlist1,cvimg2,KPlist2,sift_consensus, None,flags=2)
        cv2.imwrite('./deep-asift/temp/Affnet_homography_matches.png',img4)

    return sift_all, sift_consensus, KPlist1, KPlist2, H_sift, ET_KP, ET_M
