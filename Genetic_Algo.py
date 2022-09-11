import numpy as np
import matplotlib.pyplot as plt


class GA:
    def __init__(self, init_weight_list, init_fitness_list, number_of_generation, pop_size, learner, mutation_rate=0.2):
        self.number_of_generation = number_of_generation
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.current_generation = init_weight_list
        self.current_fitness = init_fitness_list
        self.best_gen = init_weight_list
        self.best_fitness = -1000
        self.fitness_list = []
        self.learner = learner

    def crossover(self, DNA1, DNA2):
        newDNAs = []
        for n_DNA in range(self.population_size-2):
            break_point = np.random.randint(0, len(DNA1))
            newDNA = np.append(DNA1[:break_point], DNA2[break_point:])
            newDNAs.append(self.mutation(newDNA))
        return newDNAs

    def mutation(self, DNA):
        chance = np.random.rand()
        if chance < self.mutation_rate:
            index = np.random.randint(0, len(DNA))
            DNA[index] = np.random.rand()
        return DNA

    def next_generation(self):
        # i_good_fitness = self.current_fitness.argsort()[-2:][::-1]
        i_good_fitness = list(np.array(self.current_fitness).argsort()[-2:][::-1])
        new_DNA_list = []
        new_fitness_list = []

        DNA_list = []
        for index in i_good_fitness:
            w1 = self.current_generation[0][index]
            dna_in_w = w1.reshape(w1.shape[1], -1)

            b1 = self.current_generation[1][index]
            dna_b1 = np.append(dna_in_w, b1)

            w2 = self.current_generation[2][index]
            dna_whid = w2.reshape(w2.shape[1], -1)
            dna_w2 = np.append(dna_b1, dna_whid)

            wh = self.current_generation[3][index]
            dna = np.append(dna_w2, wh)
            DNA_list.append(dna)

        DNA1 = DNA_list[0]
        DNA2 = DNA_list[1]
        new_DNA_list.append(DNA1)
        new_DNA_list.append(DNA2)

        new_DNA_list += self.crossover(DNA1=DNA1, DNA2=DNA2)

        new_input_weight = []
        new_input_bias = []
        new_hidden_weight = []
        new_output_weight = []

        for newdna in new_DNA_list:

            newdna_in_w1 = np.array(
                newdna[:self.current_generation[0][0].size])
            new_in_w = np.reshape(
                newdna_in_w1, (-1, self.current_generation[0][0].shape[1]))
            new_input_weight.append(new_in_w)

            new_in_b = np.array(
                [newdna[newdna_in_w1.size:newdna_in_w1.size+self.current_generation[1][0].size]]).T  # bias
            new_input_bias.append(new_in_b)

            sh = newdna_in_w1.size + new_in_b.size
            newdna_in_w2 = np.array(
                [newdna[sh:sh+self.current_generation[2][0].size]])
            new_hid_w = np.reshape(
                newdna_in_w2, (-1, self.current_generation[2][0].shape[1]))
            new_hidden_weight.append(new_hid_w)

            sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
            new_out_w = np.array([newdna[sl:]]).T
            new_out_w = np.reshape(
                new_out_w, (-1, self.current_generation[3][0].shape[1]))
            new_output_weight.append(new_out_w)

            fitness = self.learner.run_environment(
                new_in_w, new_in_b, new_hid_w, new_out_w)  # bias
            new_fitness_list.append(fitness)

        new_generation = [new_input_weight, new_input_bias,
                          new_hidden_weight, new_output_weight]

        return new_generation, new_fitness_list

    def show_fitness_graph(self):
        plt.plot(self.fitness_list)
        plt.xlabel('generations')
        plt.ylabel('fitness')
        plt.grid()
        print('Fitness:', self.fitness_list)
        plt.show()

    def evolve(self):
        for i in range(self.number_of_generation):
            new_generation, new_fitness = self.next_generation()
            self.current_generation = new_generation
            self.current_fitness = new_fitness
            i_max_fitness = np.argmax(self.current_fitness)
            max_fitness = self.current_fitness[i_max_fitness]
            if(max_fitness > self.best_fitness):
                self.best_fitness = max_fitness
                self.best_gen = [self.current_generation[0][i_max_fitness],self.current_generation[1][i_max_fitness],self.current_generation[2][i_max_fitness],
                    self.current_generation[3][i_max_fitness]]
            print("generation: {} ->  max_fitness: {}".format(i, max_fitness))
            self.fitness_list.append(max_fitness)
        return self.best_gen, self.best_fitness