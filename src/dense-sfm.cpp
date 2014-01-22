/* An OpenCV experiment
 * Daniel Lee, 2014
 *
 * Things to test:
 * 	calcOpticalFlowFarneback
 * 	calcOpticalFlowSF (?)
 * 	createOptFlow_DualTVL1 --> Uses an instance of DenseOpticalFlow::calc
 *
 * This is a minimally functioning dense distance map program adapted from
 * SfMToyLibrary, created by Roy Shilkrot on 1/1/12.
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

struct CloudPoint {
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
};

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts) {
	std::vector<cv::Point3d> out;
	for (unsigned int i=0; i<cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}

class IFeatureMatcher {
public:
	virtual void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches) = 0;
	virtual std::vector<cv::KeyPoint> GetImagePoints(int idx) = 0;
	virtual ~IFeatureMatcher() {};
};

class RichFeatureMatcher : public IFeatureMatcher {
private:
	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;

	std::vector<cv::Mat> descriptors;

	std::vector<cv::Mat>& imgs;
	std::vector<std::vector<cv::KeyPoint> >& imgpts;
public:
	//c'tor
	RichFeatureMatcher(std::vector<cv::Mat>& imgs,
					   std::vector<std::vector<cv::KeyPoint> >& imgpts);

	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches = NULL);

	std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};

RichFeatureMatcher::RichFeatureMatcher(std::vector<cv::Mat>& imgs_,
									   std::vector<std::vector<cv::KeyPoint> >& imgpts_) :
	imgpts(imgpts_), imgs(imgs_)
{
	detector = cv::FeatureDetector::create("PyramidFAST");
	extractor = cv::DescriptorExtractor::create("ORB");

	std::cout << " -------------------- extract feature points for all images -------------------\n";

	detector->detect(imgs, imgpts);
	extractor->compute(imgs, imgpts, descriptors);
}

void RichFeatureMatcher::MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches) {

#ifdef __SFM__DEBUG__
	const Mat& img_1 = imgs[idx_i];
	const Mat& img_2 = imgs[idx_j];
#endif
	const std::vector<cv::KeyPoint>& imgpts1 = imgpts[idx_i];
	const std::vector<cv::KeyPoint>& imgpts2 = imgpts[idx_j];
	const cv::Mat& descriptors_1 = descriptors[idx_i];
	const cv::Mat& descriptors_2 = descriptors[idx_j];

	std::vector<cv::DMatch > good_matches_,very_good_matches_;
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

	std::cout << "imgpts1 has " << imgpts1.size() << " points (descriptors " << descriptors_1.rows << ")" << std::endl;
	std::cout << "imgpts2 has " << imgpts2.size() << " points (descriptors " << descriptors_2.rows << ")" << std::endl;

	keypoints_1 = imgpts1;
	keypoints_2 = imgpts2;

	if(descriptors_1.empty()) {
		CV_Error(0,"descriptors_1 is empty");
	}
	if(descriptors_2.empty()) {
		CV_Error(0,"descriptors_2 is empty");
	}

	//matching descriptor vectors using Brute Force matcher
	cv::BFMatcher matcher(cv::NORM_HAMMING,true); //allow cross-check
	std::vector<cv::DMatch > matches_;
	if (matches == NULL) {
		matches = &matches_;
	}

	std::vector<double> dists;
	if (matches->size() == 0) {
		std::vector<std::vector<cv::DMatch> > nn_matches;
		matcher.knnMatch(descriptors_1,descriptors_2,nn_matches,1);
		matches->clear();
		for(int i=0;i<nn_matches.size();i++) {
			if(nn_matches[i].size()>0) {
				matches->push_back(nn_matches[i][0]);
				double dist = matches->back().distance;
				if(fabs(dist) > 10000) dist = 1.0;
				matches->back().distance = dist;
				dists.push_back(dist);
			}
		}
	}

	double max_dist = 0; double min_dist = 0.0;
	cv::minMaxIdx(dists,&min_dist,&max_dist);

#ifdef __SFM__DEBUG__
	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );
#endif

	std::vector<cv::KeyPoint> imgpts1_good,imgpts2_good;

	if (min_dist < 10.0) {
		min_dist = 10.0;
	}

	// Eliminate any re-matching of training points (multiple queries to one training)
	double cutoff = 4.0*min_dist;
	std::set<int> existing_trainIdx;
	for(unsigned int i = 0; i < matches->size(); i++ )
	{
		//"normalize" matching: somtimes imgIdx is the one holding the trainIdx
		if ((*matches)[i].trainIdx <= 0) {
			(*matches)[i].trainIdx = (*matches)[i].imgIdx;
		}

		int tidx = (*matches)[i].trainIdx;
		if((*matches)[i].distance > 0.0 && (*matches)[i].distance < cutoff) {
			if( existing_trainIdx.find(tidx) == existing_trainIdx.end() &&
			   tidx >= 0 && tidx < (int)(keypoints_2.size()) )
			{
				good_matches_.push_back( (*matches)[i]);
				//imgpts1_good.push_back(keypoints_1[(*matches)[i].queryIdx]);
				//imgpts2_good.push_back(keypoints_2[tidx]);
				existing_trainIdx.insert(tidx);
			}
		}
	}

	std::cout << "Keep " << good_matches_.size() << " out of " << matches->size() << " matches" << std::endl;

	*matches = good_matches_;

	return;

#if 0
#ifdef __SFM__DEBUG__
	cout << "keypoints_1.size() " << keypoints_1.size() << " imgpts1_good.size() " << imgpts1_good.size() << endl;
	cout << "keypoints_2.size() " << keypoints_2.size() << " imgpts2_good.size() " << imgpts2_good.size() << endl;

	{
		//-- Draw only "good" matches
		Mat img_matches;
		drawMatches( img_1, keypoints_1, img_2, keypoints_2,
					good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		//-- Show detected matches
		imshow( "Feature Matches", img_matches );
		waitKey(100);
		destroyWindow("Feature Matches");
	}
#endif

	vector<uchar> status;
	vector<KeyPoint> imgpts2_very_good,imgpts1_very_good;


	//Select features that make epipolar sense
	GetFundamentalMat(imgpts1_good,imgpts2_good,imgpts1_very_good,imgpts2_very_good,good_matches_);

	//Draw matches

#ifdef __SFM__DEBUG__
	{
		//-- Draw only "good" matches
		Mat img_matches;
		drawMatches( img_1, keypoints_1, img_2, keypoints_2,
					good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		//-- Show detected matches
		imshow( "Good Matches", img_matches );
		waitKey(100);
		destroyWindow("Good Matches");
	}
#endif
#endif
}


class IDistance {
public:
	virtual void OnlyMatchFeatures() = 0;
	virtual void RecoverDepthFromImages() = 0;
	virtual std::vector<cv::Point3d> getPointCloud() = 0;
	// DL: Change reference type to copy to keep from returning ref to temporary
	virtual const std::vector<cv::Vec3b> getPointCloudRGB() = 0;
	virtual ~IDistance();
};

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2)
{
	for (unsigned int i=0; i<matches.size(); i++) {
//		cout << "matches[i].queryIdx " << matches[i].queryIdx << " matches[i].trainIdx " << matches[i].trainIdx << endl;
		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
	}
}

void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps) {
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

cv::Mat GetFundamentalMat(const std::vector<cv::KeyPoint>& imgpts1,
					   const std::vector<cv::KeyPoint>& imgpts2,
					   std::vector<cv::KeyPoint>& imgpts1_good,
					   std::vector<cv::KeyPoint>& imgpts2_good,
					   std::vector<cv::DMatch>& matches
#ifdef __SFM__DEBUG__
					  ,const Mat& img_1,
					  const Mat& img_2
#endif
					  )
{
	//Try to eliminate keypoints based on the fundamental matrix
	//(although this is not the proper way to do this)
	std::vector<uchar> status(imgpts1.size());

#ifdef __SFM__DEBUG__
	std::vector< DMatch > good_matches_;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
#endif
	//	undistortPoints(imgpts1, imgpts1, cam_matrix, distortion_coeff);
	//	undistortPoints(imgpts2, imgpts2, cam_matrix, distortion_coeff);
	//
	imgpts1_good.clear(); imgpts2_good.clear();

	std::vector<cv::KeyPoint> imgpts1_tmp;
	std::vector<cv::KeyPoint> imgpts2_tmp;
	if (matches.size() <= 0) {
		imgpts1_tmp = imgpts1;
		imgpts2_tmp = imgpts2;
	} else {
		GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
	}

	cv::Mat F;
	{
		std::vector<cv::Point2f> pts1,pts2;
		KeyPointsToPoints(imgpts1_tmp, pts1);
		KeyPointsToPoints(imgpts2_tmp, pts2);
#ifdef __SFM__DEBUG__
		cout << "pts1 " << pts1.size() << " (orig pts " << imgpts1_tmp.size() << ")" << endl;
		cout << "pts2 " << pts2.size() << " (orig pts " << imgpts2_tmp.size() << ")" << endl;
#endif
		double minVal,maxVal;
		cv::minMaxIdx(pts1,&minVal,&maxVal);
		F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
	}

	std::vector<cv::DMatch> new_matches;
	std::cout << "F keeping " << cv::countNonZero(status) << " / " << status.size() << std::endl;
	for (unsigned int i=0; i<status.size(); i++) {
		if (status[i])
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			//new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
			new_matches.push_back(matches[i]);
#ifdef __SFM__DEBUG__
			good_matches_.push_back(DMatch(imgpts1_good.size()-1,imgpts1_good.size()-1,1.0));
			keypoints_1.push_back(imgpts1_tmp[i]);
			keypoints_2.push_back(imgpts2_tmp[i]);
#endif
		}
	}

	std::cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
	matches = new_matches; //keep only those points who survived the fundamental matrix

#if 0
	//-- Draw only "good" matches
#ifdef __SFM__DEBUG__
	if(!img_1.empty() && !img_2.empty()) {
		vector<Point2f> i_pts,j_pts;
		Mat img_orig_matches;
		{ //draw original features in red
			vector<uchar> vstatus(imgpts1_tmp.size(),1);
			vector<float> verror(imgpts1_tmp.size(),1.0);
			img_1.copyTo(img_orig_matches);
			KeyPointsToPoints(imgpts1_tmp, i_pts);
			KeyPointsToPoints(imgpts2_tmp, j_pts);
			drawArrows(img_orig_matches, i_pts, j_pts, vstatus, verror, Scalar(0,0,255));
		}
		{ //superimpose filtered features in green
			vector<uchar> vstatus(imgpts1_good.size(),1);
			vector<float> verror(imgpts1_good.size(),1.0);
			i_pts.resize(imgpts1_good.size());
			j_pts.resize(imgpts2_good.size());
			KeyPointsToPoints(imgpts1_good, i_pts);
			KeyPointsToPoints(imgpts2_good, j_pts);
			drawArrows(img_orig_matches, i_pts, j_pts, vstatus, verror, Scalar(0,255,0));
			imshow( "Filtered Matches", img_orig_matches );
		}
		int c = waitKey(0);
		if (c=='s') {
			imwrite("fundamental_mat_matches.png", img_orig_matches);
		}
		destroyWindow("Filtered Matches");
	}
#endif
#endif

	return F;
}

void DecomposeEssentialUsingHorn90(double _E[9], double _R1[9], double _R2[9], double _t1[3], double _t2[3]) {
	//from : http://people.csail.mit.edu/bkph/articles/Essential.pdf
#ifdef USE_EIGEN
	using namespace Eigen;

	Matrix3d E = Map<Matrix<double,3,3,RowMajor> >(_E);
	Matrix3d EEt = E * E.transpose();
	Vector3d e0e1 = E.col(0).cross(E.col(1)),e1e2 = E.col(1).cross(E.col(2)),e2e0 = E.col(2).cross(E.col(0));
	Vector3d b1,b2;

#if 1
	//Method 1
	Matrix3d bbt = 0.5 * EEt.trace() * Matrix3d::Identity() - EEt; //Horn90 (12)
	Vector3d bbt_diag = bbt.diagonal();
	if (bbt_diag(0) > bbt_diag(1) && bbt_diag(0) > bbt_diag(2)) {
		b1 = bbt.row(0) / sqrt(bbt_diag(0));
		b2 = -b1;
	} else if (bbt_diag(1) > bbt_diag(0) && bbt_diag(1) > bbt_diag(2)) {
		b1 = bbt.row(1) / sqrt(bbt_diag(1));
		b2 = -b1;
	} else {
		b1 = bbt.row(2) / sqrt(bbt_diag(2));
		b2 = -b1;
	}
#else
	//Method 2
	if (e0e1.norm() > e1e2.norm() && e0e1.norm() > e2e0.norm()) {
		b1 = e0e1.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	} else if (e1e2.norm() > e0e1.norm() && e1e2.norm() > e2e0.norm()) {
		b1 = e1e2.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	} else {
		b1 = e2e0.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	}
#endif

	//Horn90 (19)
	Matrix3d cofactors; cofactors.col(0) = e1e2; cofactors.col(1) = e2e0; cofactors.col(2) = e0e1;
	cofactors.transposeInPlace();

	//B = [b]_x , see Horn90 (6) and http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
	Matrix3d B1; B1 <<	0,-b1(2),b1(1),
						b1(2),0,-b1(0),
						-b1(1),b1(0),0;
	Matrix3d B2; B2 <<	0,-b2(2),b2(1),
						b2(2),0,-b2(0),
						-b2(1),b2(0),0;

	Map<Matrix<double,3,3,RowMajor> > R1(_R1),R2(_R2);

	//Horn90 (24)
	R1 = (cofactors.transpose() - B1*E) / b1.dot(b1);
	R2 = (cofactors.transpose() - B2*E) / b2.dot(b2);
	Map<Vector3d> t1(_t1),t2(_t2);
	t1 = b1; t2 = b2;

	cout << "Horn90 provided " << endl << R1 << endl << "and" << endl << R2 << endl;
#endif
}

bool DecomposeEtoRandT(
	cv::Mat_<double>& E,
	cv::Mat_<double>& R1,
	cv::Mat_<double>& R2,
	cv::Mat_<double>& t1,
	cv::Mat_<double>& t2)
{
#ifdef DECOMPOSE_SVD
	//Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		cout << "singular values are too far apart\n";
		return false;
	}

	Matx33d W(0,-1,0,	//HZ 9.13
		1,0,0,
		0,0,1);
	Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
#else
	//Using Horn E decomposition
	DecomposeEssentialUsingHorn90(E[0],R1[0],R2[0],t1[0],t2[0]);
#endif
	return true;
}

bool CheckCoherentRotation(cv::Mat_<double>& R) {
	std::cout << "R; " << R << std::endl;
	//double s = cv::norm(cv::abs(R),cv::Mat_<double>::eye(3,3),cv::NORM_L1);
	//std::cout << "Distance from I: " << s << std::endl;
	//if (s > 2.3) { // norm of R from I is large -> probably bad rotation
	//	std::cout << "rotation is probably not coherent.." << std::endl;
	//	return false;	//skip triangulation
	//}
	//Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> > eR(R[0]);
	//if(eR(2,0) < -0.9)
	//{
	//	cout << "rotate 180deg (PI rad) on Y" << endl;

	//	cout << "before" << endl << eR << endl;
	//	Eigen::AngleAxisd aad(-M_PI/2.0,Eigen::Vector3d::UnitY());
	//	eR *= aad.toRotationMatrix();
	//	cout << "after" << endl << eR << endl;
	//}
	//if(eR(0,0) < -0.9) {
	//	cout << "flip right vector" << endl;
	//	eR.row(0) = -eR.row(0);
	//}

	if(fabsf(determinant(R))-1.0 > 1e-07) {
		std::cerr << "det(R) != +-1.0, this is not a rotation matrix" << std::endl;
		return false;
	}

	return true;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
								   cv::Matx34d P,		//camera 1 matrix
								   cv::Point3d u1,		//homogenous image point in 2nd camera
								   cv::Matx34d P1		//camera 2 matrix
								   )
{

	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	//	cout << "u " << u <<", u1 " << u1 << endl;
	//	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u.x*P(1)-u.y*P(0);
	//	A(3) = u1.x*P1(2)-P1(0);
	//	A(4) = u1.y*P1(2)-P1(1);
	//	A(5) = u1.x*P(1)-u1.y*P1(0);
	//	Matx43d A; //not working for some reason...
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u1.x*P1(2)-P1(0);
	//	A(3) = u1.y*P1(2)-P1(1);
	cv::Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),
			  u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),
			  u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),
			  u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
			  );
	cv::Matx41d B(-(u.x*P(2,3)	-P(0,3)),
			  -(u.y*P(2,3)	-P(1,3)),
			  -(u1.x*P1(2,3)	-P1(0,3)),
			  -(u1.y*P1(2,3)	-P1(1,3)));

	cv::Mat_<double> X;
	cv::solve(A,B,X,cv::DECOMP_SVD);

	return X;
}

#define EPSILON 0.0001

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
											cv::Matx34d P,			//camera 1 matrix
											cv::Point3d u1,			//homogenous image point in 2nd camera
											cv::Matx34d P1			//camera 2 matrix
											) {
	double wi = 1, wi1 = 1;
	cv::Mat_<double> X(4,1);
	for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
		cv::Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

		//recalculate weights
		double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2)*X)(0);
		double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		cv::Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,
				  (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,
				  (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,
				  (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
				  );
		cv::Mat_<double> B = (cv::Mat_<double>(4,1) <<	  -(u.x*P(2,3)	-P(0,3))/wi,
												  -(u.y*P(2,3)	-P(1,3))/wi,
												  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
												  -(u1.y*P1(2,3)	-P1(1,3))/wi1
						  );

		cv::solve(A,B,X_,cv::DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}

double TriangulatePoints(const std::vector<cv::KeyPoint>& pt_set1,
						const std::vector<cv::KeyPoint>& pt_set2,
						const cv::Mat& K,
						const cv::Mat& Kinv,
						const cv::Mat& distcoeff,
						const cv::Matx34d& P,
						const cv::Matx34d& P1,
						std::vector<CloudPoint>& pointcloud,
						std::vector<cv::KeyPoint>& correspImg1Pt)
{
#ifdef __SFM__DEBUG__
	vector<double> depths;
#endif

//	pointcloud.clear();
	correspImg1Pt.clear();

	cv::Matx44d P1_(P1(0,0),P1(0,1),P1(0,2),P1(0,3),
				P1(1,0),P1(1,1),P1(1,2),P1(1,3),
				P1(2,0),P1(2,1),P1(2,2),P1(2,3),
				0,		0,		0,		1);
	cv::Matx44d P1inv(P1_.inv());

	std::cout << "Triangulating...";
	double t = cv::getTickCount();
	std::vector<double> reproj_error;
	unsigned int pts_size = pt_set1.size();

#if 0
	//Using OpenCV's triangulation
	//convert to Point2f
	vector<Point2f> _pt_set1_pt,_pt_set2_pt;
	KeyPointsToPoints(pt_set1,_pt_set1_pt);
	KeyPointsToPoints(pt_set2,_pt_set2_pt);

	//undistort
	Mat pt_set1_pt,pt_set2_pt;
	undistortPoints(_pt_set1_pt, pt_set1_pt, K, distcoeff);
	undistortPoints(_pt_set2_pt, pt_set2_pt, K, distcoeff);

	//triangulate
	Mat pt_set1_pt_2r = pt_set1_pt.reshape(1, 2);
	Mat pt_set2_pt_2r = pt_set2_pt.reshape(1, 2);
	Mat pt_3d_h(1,pts_size,CV_32FC4);
	cv::triangulatePoints(P,P1,pt_set1_pt_2r,pt_set2_pt_2r,pt_3d_h);

	//calculate reprojection
	vector<Point3f> pt_3d;
	convertPointsHomogeneous(pt_3d_h.reshape(4, 1), pt_3d);
	cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P(0,0),P(0,1),P(0,2), P(1,0),P(1,1),P(1,2), P(2,0),P(2,1),P(2,2));
	Vec3d rvec; Rodrigues(R ,rvec);
	Vec3d tvec(P(0,3),P(1,3),P(2,3));
	vector<Point2f> reprojected_pt_set1;
	projectPoints(pt_3d,rvec,tvec,K,distcoeff,reprojected_pt_set1);

	for (unsigned int i=0; i<pts_size; i++) {
		CloudPoint cp;
		cp.pt = pt_3d[i];
		pointcloud.push_back(cp);
		reproj_error.push_back(norm(_pt_set1_pt[i]-reprojected_pt_set1[i]));
	}
#else
	cv::Mat_<double> KP1 = K * cv::Mat(P1);
#pragma omp parallel for num_threads(1)
	for (int i=0; i<pts_size; i++) {
		cv::Point2f kp = pt_set1[i].pt;
		cv::Point3d u(kp.x,kp.y,1.0);
		cv::Mat_<double> um = Kinv * cv::Mat_<double>(u);
		u.x = um(0); u.y = um(1); u.z = um(2);

		cv::Point2f kp1 = pt_set2[i].pt;
		cv::Point3d u1(kp1.x,kp1.y,1.0);
		cv::Mat_<double> um1 = Kinv * cv::Mat_<double>(u1);
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		cv::Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);

//		cout << "3D Point: " << X << endl;
//		Mat_<double> x = Mat(P1) * X;
//		cout <<	"P1 * Point: " << x << endl;
//		Mat_<double> xPt = (Mat_<double>(3,1) << x(0),x(1),x(2));
//		cout <<	"Point: " << xPt << endl;
		cv::Mat_<double> xPt_img = KP1 * X;				//reproject
//		cout <<	"Point * K: " << xPt_img << endl;
		cv::Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

#pragma omp critical
		{
			double reprj_err = norm(xPt_img_-kp1);
			reproj_error.push_back(reprj_err);

			CloudPoint cp;
			cp.pt = cv::Point3d(X(0),X(1),X(2));
			cp.reprojection_error = reprj_err;

			pointcloud.push_back(cp);
			correspImg1Pt.push_back(pt_set1[i]);
#ifdef __SFM__DEBUG__
			depths.push_back(X(2));
#endif
		}
	}
#endif

	cv::Scalar mse = cv::mean(reproj_error);
	t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	std::cout << "Done. ("<<pointcloud.size()<<"points, " << t <<"s, mean reproj err = " << mse[0] << ")"<< std::endl;

	//show "range image"
#ifdef __SFM__DEBUG__
	{
		double minVal,maxVal;
		minMaxLoc(depths, &minVal, &maxVal);
		Mat tmp(240,320,CV_8UC3,Scalar(0,0,0)); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
		for (unsigned int i=0; i<pointcloud.size(); i++) {
			double _d = MAX(MIN((pointcloud[i].z-minVal)/(maxVal-minVal),1.0),0.0);
			circle(tmp, correspImg1Pt[i].pt, 1, Scalar(255 * (1.0-(_d)),255,255), CV_FILLED);
		}
		cvtColor(tmp, tmp, CV_HSV2BGR);
		imshow("Depth Map", tmp);
		waitKey(0);
		destroyWindow("Depth Map");
	}
#endif

	return mse[0];
}

bool TestTriangulation(const std::vector<CloudPoint>& pcloud, const cv::Matx34d& P, cv::vector<uchar>& status) {
	std::vector<cv::Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	std::vector<cv::Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());

	cv::Matx44d P4x4 = cv::Matx44d::eye();
	for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];

	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);

	status.resize(pcloud.size(),0);
	for (int i=0; i<pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = cv::countNonZero(status);

	double percentage = ((double)count / (double)pcloud.size());
	std::cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << std::endl;
	if(percentage < 0.75)
		return false; //less than 75% of the points are in front of the camera

	//check for coplanarity of points
	if(false) //not
	{
		cv::Mat_<double> cldm(pcloud.size(),3);
		for(unsigned int i=0;i<pcloud.size();i++) {
			cldm.row(i)(0) = pcloud[i].pt.x;
			cldm.row(i)(1) = pcloud[i].pt.y;
			cldm.row(i)(2) = pcloud[i].pt.z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
		cv::Vec3d x0 = pca.mean;
		double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

		for (int i=0; i<pcloud.size(); i++) {
			cv::Vec3d w = cv::Vec3d(pcloud[i].pt) - x0;
			double D = fabs(nrm.dot(w));
			if(D < p_to_plane_thresh) num_inliers++;
		}

		std::cout << num_inliers << "/" << pcloud.size() << " are coplanar" << std::endl;
		if((double)num_inliers / (double)(pcloud.size()) > 0.85)
			return false;
	}

	return true;
}

bool FindCameraMatrices(const cv::Mat& K,
						const cv::Mat& Kinv,
						const cv::Mat& distcoeff,
						const std::vector<cv::KeyPoint>& imgpts1,
						const std::vector<cv::KeyPoint>& imgpts2,
						std::vector<cv::KeyPoint>& imgpts1_good,
						std::vector<cv::KeyPoint>& imgpts2_good,
						cv::Matx34d& P,
						cv::Matx34d& P1,
						cv::vector<cv::DMatch>& matches,
						cv::vector<CloudPoint>& outCloud
#ifdef __SFM__DEBUG__
						,const Mat& img_1,
						const Mat& img_2
#endif
						)
{
	//Find camera matrices
	{
		std::cout << "Find camera matrices...";
		double t = cv::getTickCount();

		cv::Mat F = GetFundamentalMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good,matches
#ifdef __SFM__DEBUG__
								  ,img_1,img_2
#endif
								  );
		if(matches.size() < 100) { // || ((double)imgpts1_good.size() / (double)imgpts1.size()) < 0.25
			std::cerr << "not enough inliers after F matrix" << std::endl;
			return false;
		}

		//Essential matrix: compute then extract cameras [R|t]
		cv::Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

		//according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
		if(fabsf(determinant(E)) > 1e-07) {
			std::cout << "det(E) != 0 : " << determinant(E) << "\n";
			P1 = 0;
			return false;
		}

		cv::Mat_<double> R1(3,3);
		cv::Mat_<double> R2(3,3);
		cv::Mat_<double> t1(1,3);
		cv::Mat_<double> t2(1,3);

		//decompose E to P' , HZ (9.19)
		{
			if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

			if(determinant(R1)+1.0 < 1e-09) {
				//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
				std::cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << std::endl;
				E = -E;
				DecomposeEtoRandT(E,R1,R2,t1,t2);
			}
			if (!CheckCoherentRotation(R1)) {
				std::cout << "resulting rotation is not coherent\n";
				P1 = 0;
				return false;
			}

			P1 = cv::Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
						 R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
						 R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
			std::cout << "Testing P1 " << std::endl << cv::Mat(P1) << std::endl;

			std::vector<CloudPoint> pcloud,pcloud1; std::vector<cv::KeyPoint> corresp;
			double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
			double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
			std::vector<uchar> tmp_status;
			//check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
			if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
				P1 = cv::Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
				std::cout << "Testing P1 "<< std::endl << cv::Mat(P1) << std::endl;

				pcloud.clear(); pcloud1.clear(); corresp.clear();
				reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
				reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);

				if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
					if (!CheckCoherentRotation(R2)) {
						std::cout << "resulting rotation is not coherent\n";
						P1 = 0;
						return false;
					}

					P1 = cv::Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
					std::cout << "Testing P1 "<< std::endl << cv::Mat(P1) << std::endl;

					pcloud.clear(); pcloud1.clear(); corresp.clear();
					reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
					reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);

					if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
						P1 = cv::Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
						std::cout << "Testing P1 "<< std::endl << cv::Mat(P1) << std::endl;

						pcloud.clear(); pcloud1.clear(); corresp.clear();
						reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
						reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);

						if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
							std::cout << "Shit." << std::endl;
							return false;
						}
					}
				}
			}
			for (unsigned int i=0; i<pcloud.size(); i++) {
				outCloud.push_back(pcloud[i]);
			}
		}

		t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
		std::cout << "Done. (" << t <<"s)"<< std::endl;
	}
	return true;
}


class Distance : public IDistance {
private:
	std::vector<cv::KeyPoint> imgpts1,
							imgpts2,
							fullpts1,
							fullpts2,
							imgpts1_good,
							imgpts2_good;
	cv::Mat descriptors_1;
	cv::Mat descriptors_2;

	cv::Mat left_im,
			left_im_orig,
			right_im,
			right_im_orig;
	cv::Matx34d P,P1;
	cv::Mat K;
	cv::Mat_<double> Kinv;

	cv::Mat cam_matrix,distortion_coeff;

	std::vector<CloudPoint> pointcloud;
	std::vector<cv::KeyPoint> correspImg1Pt;

	bool features_matched;
public:
	std::vector<cv::Point3d> getPointCloud() { return CloudPointsToPoints(pointcloud); }
	const cv::Mat& getleft_im_orig() { return left_im_orig; }
	const cv::Mat& getright_im_orig() { return right_im_orig; }
	const std::vector<cv::KeyPoint>& getcorrespImg1Pt() { return correspImg1Pt; }
	// DL: Change return type to copy, otherwise it's reference to temporary
	const std::vector<cv::Vec3b> getPointCloudRGB() { return std::vector<cv::Vec3b>();}
		//c'tor
	Distance(const cv::Mat& left_im_, const cv::Mat& right_im_):
		features_matched(false)
	{
		left_im_.copyTo(left_im);
		right_im_.copyTo(right_im);
		left_im.copyTo(left_im_orig);
		cvtColor(left_im_orig, left_im, CV_BGR2GRAY);
		right_im.copyTo(right_im_orig);
		cvtColor(right_im_orig, right_im, CV_BGR2GRAY);

		P = cv::Matx34d(1,0,0,0,
						0,1,0,0,
						0,0,1,0);
		P1 = cv::Matx34d(1,0,0,50,
						 0,1,0,0,
						 0,0,1,0);

		cv::FileStorage fs;
		fs.open("../out_camera_data.yml",cv::FileStorage::READ);
		fs["camera_matrix"]>>cam_matrix;
		fs["distortion_coefficients"]>>distortion_coeff;

		K = cam_matrix;
		invert(K, Kinv); //get inverse of camera matrix
	}

	void OnlyMatchFeatures() {
		imgpts1.clear(); imgpts2.clear(); fullpts1.clear(); fullpts2.clear();

		std::vector<cv::Mat> imgs; imgs.push_back(left_im); imgs.push_back(right_im);
		std::vector<std::vector<cv::KeyPoint> > imgpts; imgpts.push_back(imgpts1); imgpts.push_back(imgpts2);

		RichFeatureMatcher rfm(imgs,imgpts);
		rfm.MatchFeatures(0, 1);

		imgpts1 = rfm.GetImagePoints(0);
		imgpts2 = rfm.GetImagePoints(1);

		features_matched = true;
	}

	void RecoverDepthFromImages() {

		if(!features_matched)
			OnlyMatchFeatures();

		std::vector<cv::DMatch> matches;
		FindCameraMatrices(K, Kinv, distortion_coeff, imgpts1, imgpts2, imgpts1_good, imgpts2_good, P, P1, matches, pointcloud
#ifdef __SFM__DEBUG__
						   ,left_im,right_im
#endif
						   );

		//TODO: if the P1 matrix is far away from identity rotation - the solution is probably invalid...
		//so use an identity matrix

		std::vector<cv::KeyPoint> pt_set1,pt_set2;
		GetAlignedPointsFromMatch(imgpts1,imgpts2,matches,pt_set1,pt_set2);

		TriangulatePoints(pt_set1, pt_set2, K, Kinv,distortion_coeff, P, P1, pointcloud, correspImg1Pt);
	}
};




int main(int argc, char** argv) {
    int ret_value = 0;
    const std::string binary_name(argv[0]);
    const std::string USAGE = "Usage: " + binary_name + " [frame1] [frame2]";
    if (argc != 3) {
        std::cout << USAGE << std::endl;
        return 1;
    }

    // Load images
    auto frame1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    auto frame2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat flow;

    return 0;
}
