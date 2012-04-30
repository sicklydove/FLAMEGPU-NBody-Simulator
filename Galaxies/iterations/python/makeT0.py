from simulation import Simulation
from probabilityDistribution import ProbabilityDistribution
from particleDistribution import ParticleDistribution
from particleAgent import ParticleAgent

if  __name__  ==  '__main__':

  sim=Simulation('./0.xml')
  sim.initOutput()

  distrib=ParticleDistribution(5000, False, 100, 2)
  distribMass=ProbabilityDistribution('fixed', 0.030)
  distribxPos=ProbabilityDistribution('gaussian', 0, 1.7)
  distribxVels=ProbabilityDistribution('fixed', -0.0)
  distribyVels=ProbabilityDistribution('fixed', -0.0)
  distribzVels=ProbabilityDistribution('fixed', -0.0)

  distrib.setMasses(distribMass)
  distrib.setPositions(distribxPos)
  distrib.setVelocities(distribxVels,distribyVels,distribzVels)
  sim.writeAgents(distrib.getParticleAgents())

  sim.closeOutput()
