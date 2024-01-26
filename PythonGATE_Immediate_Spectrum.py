import opengate as gate
from opengate.utility import g4_units

import uproot
import numpy as np
import matplotlib.pyplot as plt

""" Demonstration of Python GATE API

Just one pip installation away: pip install opengate
https://opengate-python.readthedocs.io

Generic Particle source into Detector --> Immediate Energy Spectrum
 """

#TODO: Add proper materials database (for us/me)
#TODO: Input option for proper Detector geometry
#(from external files maybe, as an alternative to generic slab)

output_path = 'output'

#Source Parameters
particle = "alpha"       #'e-', 'gamma', 'proton', 'neutron', 'alpha', 'ion'
Z, A = 2, 4              #Only for ions
energy = 5.5             #MeV
numParticles = 10000    
size = 2.5                #mm (Disc radius)

#Detector Parameters
size_lat = 5            #mm  
thickness = 100         #um
translation = 1         #Distance to source in cm
material = "Silicon"    #See data/GateMaterials.db

#Plot Parameters
numBins = 500
range = [energy*0.75, energy]          #MeV

def Geometry_Detector(size_lat, thickness, translation, material):
    detector = sim.add_volume("Box", "Detector")
    detector.size = [size_lat * mm, size_lat * mm, thickness * um]
    detector.translation = [0 * cm, 0 * cm, translation * cm]
    detector.material = material

def Source(particle, energy, numParticles, size):

    if particle == "ion":
        source = sim.add_source("GenericSource", "Ion")
        source.particle = f"ion {int(Z)} {int(A)}"
    else:
        source = sim.add_source("GenericSource", "Default")
        source.particle = particle

    source.position.type = "disc"
    source.position.radius = size * mm
    source.energy.mono = energy * MeV
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.n = numParticles

def Actors(detector):

    hc = sim.add_actor('DigitizerHitsCollectionActor', 'Hits')
    hc.mother = ['Detector']
    hc.output = f'{output_path}/{particle}_{energy}_hits.root'
    hc.attributes = ['TotalEnergyDeposit', 'PostPosition', 'GlobalTime', 'PreStepUniqueVolumeID']

    sc = sim.add_actor("DigitizerReadoutActor", "Singles")
    sc.output = f"{output_path}/{particle}_{energy}_singles.root"
    sc.input_digi_collection = "Hits"
    sc.group_volume = detector.name  # should be depth=1 in Gate
    sc.discretize_volume = detector.name
    sc.policy = "EnergyWeightedCentroidPosition"

def Immediate_Testplot(numBins, range):
    # Open the file
    file = uproot.open(f"output/{particle}_{energy}_singles.root")
    data = np.asarray(file['Singles']['TotalEnergyDeposit'])

    fig, ax = plt.subplots()
    ax.hist(data, bins=numBins, label=f'{numParticles} Primaries', range = range, density=True)

    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('Counts')

    if particle == "ion":
        ax.set_title(f'Energy Spectrum: {particle} Z={Z}, A={A} at {energy} MeV on {thickness} um {material}')
    else:
        ax.set_title(f'Energy Spectrum: {particle} at {energy} MeV on {thickness} um {material}')

    ax.set_xlim(range)
    ax.set_ylim([0, 1.1*max(ax.get_ylim())])

    ax.legend()

    if particle == "ion":
        plt.savefig(f'output/{particle}_{Z}_{A}_{energy}MeV_{thickness}{material}.png')
    else:
        plt.savefig(f'output/{particle}_{energy}MeV_{thickness}{material}.png')
    
    plt.show()

if __name__ == "__main__":

    sim = gate.Simulation()
    sim.volume_manager.add_material_database("data/GateMaterials.db")

    phys = sim.get_physics_user_info()
    phys.name = 'QGSP_BERT_EMV'

    #Unit Defintions
    m = g4_units.m
    cm = g4_units.cm
    keV = g4_units.keV
    MeV = g4_units.MeV
    mm = g4_units.mm
    um = g4_units.um
    Bq = g4_units.Bq

    # This is the whole simulation
    Geometry_Detector(size_lat, thickness, translation, material)
    Source(particle, energy, numParticles, size)
    Actors(detector=sim.volume_manager.volumes["Detector"])

    sim.run(start_new_process=True)

    # Plot
    Immediate_Testplot(numBins, range)