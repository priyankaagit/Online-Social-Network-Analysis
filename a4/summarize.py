import pandas as pd

def number_users(f):
    candidates = set()
    file = open('follower.txt','r').read().split('\n')
    for line in file:  
        candidates.add(line.split()[0])
    f.write("{}\t{}\n".format('Number of users collected:',str(len(candidates))))
    
    pass
    
def number_msg(f):
    messages = pd.read_csv('testtweets.csv',
                         header=None,
                         names=['polarity', 'text'])
    f.write("{}\t{}\n".format('Number of messages collected:',len(messages['polarity'])))
    
    pass

def num_cluster(f):
    cluster = []
    file = open('cluster.txt','r').read().split('\n')
    for line in file:
        cluster.append(line)
    f.write("{}\t{}\n".format('Number of communities discovered:',str(cluster[0])))
    f.write("{}\t{}\n".format('Average number of users per community:',str(cluster[1])))
    
    pass
        
def num_classify(f):
    classify = []
    file = open('classify.txt','r').read().split('\n')
    for line in file:
        classify.append(line)

    f.write("{}\t{}\n".format("Number of instance for positive sentiment", classify[0]))
    f.write("Example of tweet\n")
    f.write(classify[3])
    f.write("{}\t{}\n".format("\nNumber of instance for neutral sentiment", classify[1]))
    f.write("Example of tweet\n")
    f.write(classify[4])
    f.write("{}\t{}\n".format("\nNumber of instance for negative sentiment", classify[2]))
    f.write("Example of tweet\n")
    f.write(classify[5])
    
    pass

def main():
    f = open('summary.txt','w')
    number_users(f)
    number_msg(f)
    num_cluster(f)
    num_classify(f)
    f.close()
    print('Script complete, check summary.txt')

if __name__ == '__main__':
    main()
