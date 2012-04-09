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
      self.distribType='fixed'
      self.fixedVal=val1

    elif(distribType is 'circle'):
      self.distribType='circle'
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

  def getItem(self):
    if(self.distribType is 'gaussian'):
      return random.gauss(self.mu, self.sigma)

    elif(self.distribType is 'linear'):
      return random.uniform(self.minVal, self.maxVal)

    elif(self.distribType is 'fixed'):
      return self.fixedVal

    elif(self.distribType is 'circle'):
      randRadius=random.uniform(0, self.radius)
      randTheta=random.uniform(0, 2*math.pi)
      randThetaTwo=random.uniform(0,math.pi)
      
      xPos=randRadius*(math.cos(randTheta)*math.sin(randThetaTwo))+self.centre[0]
      yPos=randRadius*(math.sin(randTheta)*math.sin(randThetaTwo))+self.centre[1]
      zPos=randRadius*math.cos(randThetaTwo)+self.centre[2]

      return (xPos,yPos,zPos)


#archive code...
    elif(self.distribType is 'oldCircle'):
      randRadius=random.uniform(0, self.radius)
      randxPos=random.uniform(self.centre[0]-randRadius, self.centre[0]+randRadius)

      radiusSq=randRadius**2
      xPosSq=(randxPos-self.centre[0])**2

      maxY=quadratic(1, 0-(2*self.centre[1]), ((self.centre[1]**2)-(radiusSq-xPosSq))[0])

      randyPos=random.uniform(self.centre[1], maxY)
      randyPosSq=(randyPos-self.centre[1])**2

      zMinusC=sqrt(radiusSq-xPosSq-randyPosSq)

      #invert positions
      if(random.random()>0.5):
        randyPos=self.centre[1]+(self.centre[1]-randyPos)
        zPos=zMinusC+self.centre[2]

      else:
        zPos=self.centre[2]-zMinusC

      return (randxPos,randyPos,zPos)
