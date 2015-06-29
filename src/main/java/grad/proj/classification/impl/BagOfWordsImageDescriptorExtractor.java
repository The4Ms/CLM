package grad.proj.classification.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;


/**
 * port of BOWImgDescriptorExtractor of OpenCV
 * 
 */
class BagOfWordsImageDescriptorExtractor {
	
	private Mat vocabulary;
	private DescriptorExtractor dextractor;
	private DescriptorMatcher dmatcher;
	
	public BagOfWordsImageDescriptorExtractor( DescriptorExtractor _dextractor, DescriptorMatcher _dmatcher ) {
		dextractor = _dextractor;
		dmatcher = _dmatcher;
	}
	
	void setVocabulary(Mat _vocabulary )
	{
		dmatcher.clear();
		vocabulary = _vocabulary;
		dmatcher.add(Arrays.asList(vocabulary));
	}
	
	Mat getVocabulary()
	{
		return vocabulary;
	}
	
	void compute(Mat image, MatOfKeyPoint keypoints, Mat imgDescriptor,
	             List<List<Integer>> pointIdxsOfClusters, Mat _descriptors)
	{
		imgDescriptor.release();
		
		if(keypoints.empty())
			return;
		
		int clusterCount = descriptorSize(); // = vocabulary.rows
		
		// Compute descriptors for the image.
		Mat descriptors = new Mat();
		dextractor.compute( image, keypoints, descriptors );
		
		// Match keypoint descriptors to cluster center (to vocabulary)
		MatOfDMatch matches = new MatOfDMatch();
		dmatcher.match( descriptors, matches);
		
		// Compute image descriptor
		if(pointIdxsOfClusters != null)
		{
			pointIdxsOfClusters.clear();
			for(int i=0; i<clusterCount; i++)
				pointIdxsOfClusters.add(new ArrayList<Integer>());
		}
		
		Mat imgDescriptorTemp = new Mat( 1, clusterCount, descriptorType(), new Scalar(0.0));
		float[] dptr = new float[(int) (imgDescriptorTemp.total() * imgDescriptorTemp.channels())];
		
		imgDescriptorTemp.get(0, 0, dptr);
		
		Iterator<DMatch> matchesIter = matches.toList().iterator();
		while(matchesIter.hasNext()){
			DMatch matchesOfI = matchesIter.next();
		
			int queryIdx = matchesOfI.queryIdx;
			int trainIdx = matchesOfI.trainIdx; // cluster index
			//CV_Assert( queryIdx == (int)i );
			
			dptr[trainIdx] = dptr[trainIdx] + 1.f;
			
			if( pointIdxsOfClusters != null ){
				pointIdxsOfClusters.get(trainIdx).add(queryIdx);
			}
		}
		
		imgDescriptorTemp.put(0, 0, dptr);
		
		// Normalize image descriptor.
//		Mat rowsConstAsMat = new Mat(1, clusterCount, descriptorType(), new Scalar(1.0f / descriptors.rows()));
//		imgDescriptorTemp.mul(rowsConstAsMat).copyTo(imgDescriptor);;
		imgDescriptorTemp.copyTo(imgDescriptor);
		
		// Add the descriptors of image keypoints
		if (_descriptors != null) {
			descriptors.assignTo(_descriptors);
		}
	}
	
	int descriptorSize()
	{
	return vocabulary == null ? 0 : vocabulary.rows();
	}
	
	int descriptorType()
	{
	return CvType.CV_32FC1;
	}
	
}