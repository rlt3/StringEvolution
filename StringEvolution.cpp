#include <cstdio>
#include <string>
#include <random>
#include <vector>
#include <algorithm>
#include "arg.h"

static inline int
minimum (int a, int b, int c)
{
    return std::min(std::min(a, b), c);
}

class Chromosome {
public:
    std::string genes;
    double fitness_value;

    /* initializer constructor */
    Chromosome ()
        : fitness_value(0)
        , genes("")
    { }

    /* copy constructor */
    Chromosome (const Chromosome &c)
    {
        this->genes = c.genes;
        this->fitness_value = c.fitness_value;
    }

    Chromosome (int max_genes, bool constant_width, std::mt19937_64 &rand)
        : fitness_value(0)
        , genes("")
    {
        std::uniform_int_distribution<int> length(1, max_genes);
        const int num_genes = constant_width? max_genes : length(rand);
        char buff[num_genes + 1];
        int i;

        for (i = 0; i < num_genes; i++) {
            buff[i] = Chromosome::RandomGene(rand);
        }

        buff[i] = '\0';
        this->genes = std::string(buff);
    }

    /*
     * Levenshtein Distance. This enables us to do fitness with variable 
     * length strings.
     * https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_full_matrix
     */
    double
    fitness (const std::string &target)
    {
        const int m = this->genes.length();
        const int n = target.length();
        int d[m + 1][n + 1];
        int substitution_cost;
        int i, j;

        /* initialize everything */
        for (j = 0; j <= n; j++)
            for (i = 0; i <= m; i++)
                d[i][j] = 0;

        /* on each axis setup the index of each character as a value */
        for (i = 1; i <= m; i++)
            d[i][0] = i;
        for (j = 1; j <= n; j++)
            d[0][j] = j;

        /*
         * We loop from 1 because we need to leave each axis untouch (above).
         * We are filling in the insides of the matrix and determining the
         * minimum substituion cost to have the two strings match at each
         * specific index.
         */
        for (j = 1; j <= n; j++) {
            for (i = 1; i <= m; i++) {
                if (this->genes[i - 1] == target[j - 1])
                    substitution_cost = 0;
                else
                    substitution_cost = 1;
                d[i][j] = minimum(d[i-1][j] + 1,
                                  d[i][j-1] + 1,
                                  d[i-1][j-1] + substitution_cost);
            }
        }

        this->fitness_value = 1.0 / (double) d[m][n];
        return this->fitness_value;
    }

    static int
    RandomGene (std::mt19937_64 &rand)
    {
        /* disribution in the ASCII range */
        static std::uniform_int_distribution<int> character(32, 126);
        return character(rand);
    }
};

bool chromosome_compare (const Chromosome &a, const Chromosome &b)
{
    /* sort in descending order */
    return a.fitness_value > b.fitness_value;
}

struct PopulationConfig {
    int population_size;
    double elite_rate;
    double cull_thresh;
    double mutate_rate;
    double crossover_rate;
    int gen_limit;
    bool quiet;
    bool constant_width;
    std::string target;
};

class Population {
public:
    Population (PopulationConfig config, std::mt19937_64 &rand)
        : config(config)
        , rand(rand)
    {
        const unsigned int num_genes = config.target.size();
        const unsigned int constant_width = config.constant_width;
        chromosomes.reserve(config.population_size);
        next_generation.reserve(config.population_size);
        for (int i = 0; i < config.population_size; i++)
            chromosomes.push_back(Chromosome(num_genes, constant_width, rand));
        this->fitness();
    }

    Chromosome
    best ()
    {
        return chromosomes.front();
    }

    bool
    reached_target ()
    {
        return std::isinf(this->best().fitness_value);
    }

    void
    advance ()
    {
        static std::uniform_real_distribution<double> range(0.0, 1.0);
        const int num_elite = config.elite_rate * config.population_size;
        const int threshold = config.cull_thresh * chromosomes.size();
        Chromosome father, mother, child;

        /* the top performers (elite) automatically go to next generation */
        next_generation.clear();
        for (int i = 0; i < num_elite; i++)
            next_generation.push_back(chromosomes[i]);

        father = next_generation.front();
        while (next_generation.size() < config.population_size) {
            mother = this->select(threshold);

            if (range(rand) < config.crossover_rate) {
                child = this->crossover(father, mother);
            }

            if (range(rand) < config.mutate_rate) {
                child = this->mutate(mother);
            }

            next_generation.push_back(child);
        }

        chromosomes = next_generation;
        this->fitness();
    }

protected:
    std::vector<Chromosome> chromosomes;
    std::vector<Chromosome> next_generation;
    const PopulationConfig config;
    std::mt19937_64 &rand;

    /* fitness over the entire population and sort chromosomes by most fit */
    void
    fitness () 
    {
        for (auto &c : chromosomes)
            c.fitness(config.target);
        std::sort(chromosomes.begin(), chromosomes.end(), chromosome_compare);
    }

    /* crossover two chromosomes */
    Chromosome
    crossover (Chromosome c1, Chromosome c2)
    {
        /* locus can only be us to the length of the smallest string */
        static std::uniform_int_distribution<int> direction(0, 1);
        const int max_locus = std::min(c1.genes.length(), c2.genes.length());
        std::uniform_int_distribution<int> range(0, max_locus);
        const int locus = range(rand);
        Chromosome child;

        /* this crossover can modify length of string */
        if (direction(rand) == 1) {
            child.genes = c1.genes.substr(0, locus) + c2.genes.substr(locus);
            return c1;
        } else {
            child.genes = c2.genes.substr(0, locus) + c1.genes.substr(locus);
        }

        return c1;
    }

    /* mutate a the chromosome with a random gene */
    Chromosome
    mutate (Chromosome c)
    {
        static std::uniform_int_distribution<int> assign_task(0, 2);
        std::uniform_int_distribution<int> gene_range(0, c.genes.length());
        const int index = gene_range(rand);
        const int gene = Chromosome::RandomGene(rand);
        /* if we have a constant width then only do an in-place change */
        const int task = config.constant_width? 2 : assign_task(rand);

        switch (task) {
        /* inserting gene at the index, increasing size of chromosome */
        case 0:
            c.genes.insert(index, 1, gene);
            break;

        /* deleting gene at the index, decreasing size */
        case 1:
            c.genes.erase(index, 1);
            break;

        /* changing gene in place */
        default:
        case 2:
            c.genes[index] = gene;
            break;
        }

        return c;
    }

    /* select a random chromosome but not one beyond threshold */
    Chromosome
    select (const int threshold)
    {
        std::uniform_int_distribution<int> range(0, threshold);
        return chromosomes[range(rand)];
    }
};

void
usage (const char *prog)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS] <target-string>\n"
            "OPTIONS:\n"
            "  -e [0.0 - 1.0]\n"
            "    The elitism threshold for the population. This percent of\n"
            "    the population are automatically added to next generaiton.\n\n"
            "  -t [0.0 - 1.0]\n"
            "    The cull threshold. The percentage of the population you\n"
            "    would like to keep between generations, e.g. 0.6 keeps the\n"
            "    best 60%% of the population.\n\n"
            "  -m [0.0 - 1.0]\n"
            "    The rate at which chromosomes are mutated.\n\n"
            "  -c [0.0 - 1.0]\n"
            "    The rate at which chromosomes are crossed-over.\n\n"
            "  -p <number>\n"
            "    The population or number of chromosomes each generation.\n\n"
            "  -l <number>\n"
            "    Limit number of generations evolved to the number given.\n\n"
            "  -q\n"
            "    Quiet the output. Just output the best of each generation.\n\n"
            "  -w\n"
            "    Create all chromosomes with the same width as target string.\n"
            , prog);
    exit(1);
}

PopulationConfig
args_get_config (int argc, char **argv)
{
    PopulationConfig config;
    char *prog = argv[0];
    char *argv0; /* for ARGBEGIN macro, see arg.h */

    if (argc < 2)
        usage(prog);

    /* sane defaults */
    config.population_size = 1000;
    config.elite_rate      = 0.10;
    config.cull_thresh     = 0.70;
    config.mutate_rate     = 0.25;
    config.crossover_rate  = 0.50;
    config.gen_limit       = 0;
    config.quiet           = false;
    config.constant_width  = false;
    config.target = "Leroy is the coolest!"; /* important, don't change */

    ARGBEGIN {
        case 'e': config.elite_rate = atof(ARGF()); break;
        case 't': config.cull_thresh = atof(ARGF()); break;
        case 'm': config.mutate_rate = atof(ARGF()); break;
        case 'c': config.crossover_rate = atof(ARGF()); break;
        case 'p': config.population_size = atoi(ARGF()); break;
        case 'l': config.gen_limit = atoi(ARGF()); break;
        case 'q': config.quiet = true; break;
        case 'w': config.constant_width = true; break;
        case 'h': usage(prog); break;
        default: break;
    } ARGEND

    config.target = std::string(argv[argc - 1]);

    if (!config.quiet) {
        printf("population     = %0.2d\n"
               "elite_rate     = %0.2lf\n"
               "cull_thresh    = %0.2lf\n"
               "mutate_rate    = %0.2lf\n"
               "crossover_rate = %0.2lf\n"
               "target         = %s\n",
               config.population_size, config.elite_rate, config.cull_thresh,
               config.mutate_rate, config.crossover_rate, config.target.c_str());
    }

    return config;
}

int
main (int argc, char **argv)
{
    PopulationConfig config;
    std::random_device device;
    std::mt19937_64 rand(device());

    config = args_get_config(argc, argv);
    Population pop(config, rand);

    for (int i = 0 ;; i++) {
        printf("%s", pop.best().genes.c_str());
        if (!config.quiet)
            printf(" %lf", pop.best().fitness_value);
        putchar('\n');
        if (pop.reached_target()) {
            if (!config.quiet)
                printf("Reached target fitness in %d generations\n", i);
            break;
        }
        if (config.gen_limit > 0 && i >= config.gen_limit)
            break;
        pop.advance();
    }

    return 0;
}
