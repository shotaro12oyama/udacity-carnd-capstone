from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):

        PATH_TO_GRAPH = r'light_classification/models/frozen_inference_graph0517.pb'
        self.graph = tf.Graph()
        self.threshold = .3

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        
        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            scores, classes = self.sess.run(
                [self.scores, self.classes],
                feed_dict={self.image_tensor: img_expand})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        rospy.loginfo('classification score:{}, class:{}'.format(scores[0], classes[0]))

        if scores[0] > self.threshold:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
