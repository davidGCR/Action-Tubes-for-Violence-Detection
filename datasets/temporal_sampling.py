import random
import numpy as np
from operator import itemgetter

def crop(frames, start, size, stride):
    # todo more efficient
    # padding by loop
    while start + (size - 1) * stride > len(frames) - 1:
        frames *= 2
    return frames[start:start + (size - 1) * stride + 1:stride]

class RandomCrop(object):
    def __init__(self, size, stride=1, input_type="rgb"):
        """Get random temporal segment

        Args:
            size (int): Temporal length of the segment
            stride (int, optional): Stride for sampling. Defaults to 1.
            input_type (str, optional): Type of inputs to sample. Defaults to "rgb".
        """
        self.size = size
        self.stride = stride
        self.input_type = input_type

    def __call__(self, frames):
        start = random.randint(
            0, max(0,
                   len(frames) - 1 - (self.size - 1) * self.stride)
        )
        crop_r = crop(frames, start, self.size, self.stride)
        if self.input_type == "rgb":
          return crop_r
        elif self.input_type == "dynamic-images":
          return [crop_r]
    
class CenterCrop(object):
    """Get middle temporal segment

        Args:
            size (int): Temporal length of the segment
            stride (int, optional): Stride for sampling. Defaults to 1.
            input_type (str, optional): Type of inputs to sample. Defaults to "rgb".
        """
    def __init__(self, size, stride=1, input_type="rgb"):
        self.size = size
        self.stride = stride
        self.input_type = input_type

    def __call__(self, frames):
        start = max(0, len(frames) // 2 - self.size * self.stride // 2)
        crop_r = crop(frames, start, self.size, self.stride)
        if self.input_type == "rgb":
          return crop_r
        elif self.input_type == "dynamic-images":
          return [crop_r] 

class SegmentsCrop(object):
    def __init__(self, size, segment_size=15, stride=1, overlap=0.5, padding=True, position="start"):
        """
        Args:
            size (int): number of segments
            segment_size (int): length of each segment
            stride (int): frames to skip into a segment
            overlap (float): overlapping between each segment
            padding (bool): cut or add segments to get 'size' segments for each sample
            position (str): In case we want one segmet from the beginning, middle and end of the video 
        """
        self.size = size
        self.segment_size = segment_size
        self.stride = stride
        self.overlap = overlap
        self.overlap_length = int(self.overlap*self.segment_size)
        self.padding = padding
        self.position = position
    
    def __remove_short_segments__(self, segments):
        segments = [s for s in segments if len(s)==self.segment_size]
        return segments
    
    def __padding__(self, segment_list):
        if  len(segment_list) < self.size:
            idx = len(segment_list) - 1
            last_element = segment_list[idx]
            # while len(last_element) != self.segment_size and i >=0:
            #     i -= 1
            #     last_element = segment_list[i]

            for i in range(self.size - len(segment_list)):
                segment_list.append(last_element)
        # elif len(segment_list) > self.size:
        #     segment_list = segment_list[0:self.size]
        return segment_list
    
    def __choose_one_position__(self, segments):
        if self.position == "start":
            return segments[0]
        elif self.position == "middle":
            return segments[int(len(segments)/2)]
        elif self.position == "random":
            start = random.randint(0, len(segments)-1)
            while len(segments[(start)]) != self.segment_size:
                start = random.randint(0, len(segments)-1)
            
            # print(segments[start], len(segments))
            return segments[(start)]
    
    def __choose_multiple_positions__(self, segments):
        if self.position == "start":
            return segments[0:self.size]
        elif self.position == "middle":
            # segments = [s for s in segments if len(s)==self.size]
            c = int(len(segments)/2)
            start = c-int(self.size/2)
            end = c+int(self.size/2)
            idxs = [i for i in range(start,end+1)] if end-start < self.size else [i for i in range(start,end)]
            segments = list(itemgetter(*idxs)(segments))
            return segments
        elif self.position == "random":
            # segments = [s for s in segments if len(s)==self.size]
            segments = random.sample(segments, self.size)
            segments.sort()
            return segments

    def __call__(self, frames):
        
        indices = [x for x in range(0, len(frames), self.stride)]
        
        indices_segments = [indices[x:x + self.segment_size] for x in range(0, len(indices), self.segment_size-self.overlap_length)]

        indices_segments = self.__remove_short_segments__(indices_segments)
        
        if len(indices_segments)<self.size and self.padding:
            indices_segments = self.__padding__(indices_segments)
        if self.size == 1:
            indices_segments = [self.__choose_one_position__(indices_segments)]
        elif self.size > 1:
            indices_segments = self.__choose_multiple_positions__(indices_segments)
        elif self.size == 0: #all
            indices_segments = indices_segments
           
        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            video_segments.append(segment)
        # print(video_segments)
        return video_segments

if __name__ == '__main__':
    # temp_transform = CenterCrop(size=16, stride=2)
    # temp_transform = RandomCrop(size=16, stride=2)
    
    temp_transform = SegmentsCrop(size=4, segment_size=16, stride=1,overlap=0.6,padding=True, position='middle')
    
    frames = list(range(1, 40))
    frames = temp_transform(frames)
    print(frames)
    print(len(frames))