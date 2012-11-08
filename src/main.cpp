#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include "jacobian.h"

using namespace std;

class ExtrinsicOptimizer : public ceres::CostFunction
{
public:
  explicit ExtrinsicOptimizer(cv::Mat &K,cv::Mat &world,cv::Mat &x2d,int num_obs):num_obs_(num_obs)
  {
	K.copyTo(K_);
	world.copyTo(world_);	//! 3dpts.
	set_num_residuals(2*num_obs);
	mutable_parameter_block_sizes()->push_back(6);

	x2d.copyTo(x2d_);

	fx = K_.at<double>(0,0);
	fy = K_.at<double>(1,1);
	cx = K_.at<double>(0,2);
	cy = K_.at<double>(1,2);
  }
  virtual bool Evaluate(double const* const* parameters,
			double *residuals,
			double **jacobians) const{
  	// parameters wx,wy,wz,tx,ty,tz;
	// X Y Z
	double wx = parameters[0][0];
	double wy = parameters[0][1];
	double wz = parameters[0][2];
	double tx = parameters[0][3];
	double ty = parameters[0][4];
	double tz = parameters[0][5];

	cv::Mat R(3,3,CV_64F);
	cv::Mat t(3,1,CV_64F);
	cv::Mat P(3,4,CV_64F);
	cv::Mat w(3,1,CV_64F);

	w.at<double>(0,0) = wx;
	w.at<double>(1,0) = wy;
	w.at<double>(2,0) = wz;

	t.at<double>(0,0) = tx;
	w.at<double>(1,0) = ty;
	w.at<double>(2,0) = tz;
	
	ceres::AngleAxisRotationMatrix((double*)w.ptr(),(double*)R.ptr());

	R.copyTo(P(cv::Range(0,3),cv::Range(0,3)));
        t.copyTo(P(cv::Range(0,3),cv::Range(3,4)));

	P = K_*P;	//! projection matrix

	for(int i=0;i<num_obs_;i++)
	{
		double X = world_.at<double>(0,i);
		double Y = world_.at<double>(1,i);
		double Z = world_.at<double>(2,i);

		cv::Mat image = P*world_.col(i);	

		for(unsigned int j=0;j<N;j++)
        	{
     	           	image.at<double>(0,j) /= image.at<double>(2,j);
        	        image.at<double>(1,j) /= image.at<double>(2,j);
                	image.at<double>(2,j) /= image.at<double>(2,j);
		}

		residuals[i*2 + 0] = x2d_.at<double>(0,i) - image.at<double>(0,i);
		residuals[i*2 + 1] = x2d_.at<double>(1,i) - image.at<double>(1,i);

		jacobians[i*2 + 0][0] = JX0;
		jacobians[i*2 + 0][1] = JX1;
		jacobians[i*2 + 0][2] = JX2;
		jacobians[i*2 + 0][3] = JX3;
		jacobians[i*2 + 0][4] = JX4;
		jacobians[i*2 + 0][5] = JX5;
		jacobians[i*2 + 1][0] = JY0;
		jacobians[i*2 + 1][1] = JY1;
		jacobians[i*2 + 1][2] = JY2;
		jacobians[i*2 + 1][3] = JY3;
		jacobians[i*2 + 1][4] = JY4;
		jacobians[i*2 + 1][5] = JY5;
	}
  }
private:
  int nums;
  double fx,fy,cx,cy;
  int num_obs_;
  cv::Mat world_; //! 4xN world 3D point matrix.
  cv::Mat K_; //! intrinsic parameter
  cv::Mat x2d_; //! 2d image pts
};

double randn(void) {
  static double V1, V2, S;
  static int phase = 0;
  double X;



  if (phase == 0) {

    do {
      double U1 = (double)rand() / RAND_MAX;
      double U2 = (double)rand() / RAND_MAX;
      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
    } while (S >= 1 || S == 0);

    X = V1 * sqrt(-2 * log(S) / S);

  } else X = V2 * sqrt(-2 * log(S) / S);

  phase = 1 - phase;
  return X;
}

int main(int argc, char** argv)
{
	google::InitGoogleLogging(argv[0]);
	const unsigned int N = 10;	//! how many data will be generated	

	//! prepare initial parameters
	cv::Mat Euler(3,1,CV_64F);
	Euler.at<double>(0,0) = 30.0;	//! pitch
	Euler.at<double>(1,0) = 20.0;	//! roll
	Euler.at<double>(2,0) = 10.0;	//! yaw

	//! Assume intrinsic matrix as Identity 
	cv::Mat K(3,3,CV_64F);
	K.at<double>(0,0) = 1.0;
	K.at<double>(0,1) = 0.0;
	K.at<double>(0,2) = 0.0;
	K.at<double>(1,0) = 0.0;
	K.at<double>(1,1) = 1.0;
	K.at<double>(1,2) = 0.0;
	K.at<double>(2,0) = 0.0;
	K.at<double>(2,1) = 0.0;
	K.at<double>(2,2) = 1.0;

	cv::Mat t(3,1,CV_64F);		//! translation vector
	t.at<double>(0,0) = 1.0;
	t.at<double>(1,0) = 2.0;
	t.at<double>(2,0) = 3.0;

	cv::Mat R(3,3,CV_64F);		//! rotation matrix
	cv::Mat w(3,1,CV_64F);		//! angle-axis vector
	cv::Mat world(4,N,CV_64F);	//! 3d world point
	cv::Mat image(3,N,CV_64F);	//! 2d image point
	cv::Mat P(3,4,CV_64F);

	//! Prepare ground truth data
	ceres::EulerAnglesToRotationMatrix((double*)Euler.ptr(),3,(double*)R.ptr());
	cout << "Preparing ground truth R\n";
	cout << R << endl;

	//! R to angle axis
	ceres::RotationMatrixToAngleAxis((double*)R.ptr(),(double*)w.ptr());
	cout << "Ground truth angle axis\n";
	cout << w << endl;

	//! Project world image to image plane using P = K[R t]
	cout << "Projection matrix\n";
	R.copyTo(P(cv::Range(0,3),cv::Range(0,3)));
	t.copyTo(P(cv::Range(0,3),cv::Range(3,4)));
	cout << P << endl;

	for(unsigned int i=0;i<N;i++)
	{
		world.at<double>(0,i) = randn();
		world.at<double>(1,i) = randn();
		world.at<double>(2,i) = randn();
		world.at<double>(3,i) = 1.0;
	}

	image = P*world;

	for(unsigned int i=0;i<N;i++)
	{
		image.at<double>(0,i) /= image.at<double>(2,i);
		image.at<double>(1,i) /= image.at<double>(2,i);
		image.at<double>(2,i) /= image.at<double>(2,i);
	}

	/// random perturbation
	for(unsigned int i=0;i<N;i++)
	{
		image.at<double>(0,i) += 0.2*randn();
		image.at<double>(1,i) += 0.2*randn();
		image.at<double>(2,i) += 0.2*randn();
	}

	w.at<double>(0,0) = 0.2*randn();
	w.at<double>(1,0) = 0.2*randn();
	w.at<double>(2,0) = 0.2*randn();

	t.at<double>(0,0) = 0.2*randn();
	t.at<double>(1,0) = 0.2*randn();
	t.at<double>(2,0) = 0.2*randn();

	cv::Mat param(6,1,CV_64F);
	param.at<double>(0,0) = w.at<double>(0,0);
	param.at<double>(1,0) = w.at<double>(1,0);
	param.at<double>(2,0) = w.at<double>(2,0);
	param.at<double>(3,0) = t.at<double>(0,0);
	param.at<double>(4,0) = t.at<double>(1,0);
	param.at<double>(5,0) = t.at<double>(2,0);
	
	ceres::Problem problem;
	problem.AddResidualBlock(new ExtrinsicOptimizer(K,world,image,N),NULL,(double*)param.ptr());
	return 0;
}
