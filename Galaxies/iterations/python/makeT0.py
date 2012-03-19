import sys
sys.dont_write_bytecode=True
#import particleAgent
from simulation import Simulation 
from probabilityDistribution import ProbabilityDistribution
from particleDistribution import ParticleDistribution
import particleAgent

if  __name__  ==  '__main__':

  simulation=Simulation('./0.xml')
  simulation.initOutput()

  distrib=ParticleDistribution(4000, False)
  distribMass=ProbabilityDistribution('fixed', 0.0)
  distribPos=ProbabilityDistribution('linear', -2, 2)
  distribVels=ProbabilityDistribution('linear', -0.0,0.0)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  simulation.writeAgents(distrib.getParticleAgents())

  distrib=ParticleDistribution(1000, True)
  distribMass=ProbabilityDistribution('fixed', 0.03)
  distribPos=ProbabilityDistribution('circle', (-2,-1,-2), 0.2)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  simulation.writeAgents(distrib.getParticleAgents())

  distrib=ParticleDistribution(1000, True)
  distribMass=ProbabilityDistribution('fixed', 0.03)
  distribPos=ProbabilityDistribution('circle', (1,1,-1.2), 0.2)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  simulation.writeAgents(distrib.getParticleAgents())

  distrib=ParticleDistribution(1000, True)
  distribMass=ProbabilityDistribution('fixed', 0.03)
  distribPos=ProbabilityDistribution('circle', (0.3,-0.8,-.9), 0.2)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  simulation.writeAgents(distrib.getParticleAgents())

  simulation.closeOutput()
