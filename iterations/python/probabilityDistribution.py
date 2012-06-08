import random
import particleAgent
import math
class ProbabilityDistribution:
  def __init__(self, distribType, val1, val2=None):
    if(distribType is ('gaussian' or 'normal')):
      self.distribType='gaussian'
      if(val2 is None):
        print "ERROR: Must define a mu and sigma value for a Gaussian distribution"
        exit()
      else:
        self.mu=val1
        self.sigma=val2

    elif(distribType is ('fixed')):
      if(val2):
        print "Error: cannot define multiple variables in a fixed distribution"
       	exit()
      else:
        self.distribType='fixed'
        self.fixedVal=val1

    elif(distribType is 'sphere'):
      self.distribType='sphere'
      if(val2 is None):
        print "Error: Must define a centre (x,y,z) term and radius for a spherical distribution"
       	exit()
      else:
        self.centre=val1
        self.radius=val2

    else:
      self.distribType='linear'
      if(val2 is None):
        print "ERROR: Must define a Min and Max value for a random distribution"
        exit()
      else:
        self.minVal=val1
        self.maxVal=val2

  def getType(self):
    return self.distribType

  def getRadius(self):
    return self.radius

  def getItem(self):
    if(self.distribType is 'gaussian'):
      return random.gauss(self.mu, self.sigma)

    elif(self.distribType is 'linear'):
      return random.uniform(self.minVal, self.maxVal)

    elif(self.distribType is 'fixed'):
      return self.fixedVal

    elif(self.distribType is 'sphere'):
      randRadius=random.uniform(0, self.radius)
      randTheta=random.uniform(0, 2*math.pi)
      randThetaTwo=random.uniform(0,math.pi)
      
      #Define positions with parametric form instead of randomising one param and solving quadratic
      xPos=randRadius*(math.cos(randTheta)*math.sin(randThetaTwo))+self.centre[0]
      yPos=randRadius*(math.sin(randTheta)*math.sin(randThetaTwo))+self.centre[1]
      zPos=randRadius*math.cos(randThetaTwo)+self.centre[2]

      return (xPos,yPos,zPos)
