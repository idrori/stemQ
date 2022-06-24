import openai
import umap
import json
import matplotlib.pyplot as plt
import os

openai.api_key = "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"

image_location = "UMAP.png"
embedding_engine = 'text-similarity-babbage-001'

def make_embeddings(embedding_engine, courses):
    for course in courses:
        list_of_embeddings = []
        print("Currently embedding " + course + "...")
        folder = './Data/' + course + '/'
        for num in range(1, len(os.listdir(folder)) + 1):
            if num < 10:
                q_num = '0' + str(num)
            else:
                q_num = str(num)
            json_location = folder + course + '_Question_' + q_num + '.json'
            with open(json_location, 'r') as f:
                data = json.load(f)
            raw_question = data['Original question']
            embedding = openai.Embedding.create(input = raw_question,
                                                engine = embedding_engine)['data'][0]['embedding']
            list_of_embeddings.append(embedding)
        embeddings = {'list_of_embeddings':list_of_embeddings}
        if not os.path.isdir('./embeddings'):
            os.mkdir('./embeddings')
        with open('./embeddings/' + course + ' embeddings.json', 'w') as f:
            f.write(json.dumps(embeddings))

def get_embeddings(embeddings_file):
    """
    Retrieves embeddings from embeddings_file. Embeddings are assumed to be (n x d).
    """
    with open(embeddings_file, 'r') as f:
        points = json.load(f)['list_of_embeddings']
    return points


def reduce_via_umap(embeddings, num_dims=2):
    """
    Reduces the dimensionality of the provided embeddings(which are vectors) to num_dims via UMAP.
    If embeddings was an (n x d) numpy array, it will be reduced to a (n x num_dims) numpy array.
    """
    reducer = umap.UMAP(n_components=num_dims)
    reduced = reducer.fit_transform(embeddings)
    return reduced

def plot_clusters(points, image_loc, show=False, question_labels=False, label_font='xx-small',
                  dpi=200, width=9.5, height=6.5, legend_loc=(1, 1.01), right_shift=0.72):
    """
    Plots clusters of points. points is assumed to be a n by 2 numpy array.
    Set question_labels to True if you want to see each point labeled with its question number.
    Set show to True if you want the created plot to pop up.
    The other parameters are defaulted to values that we have found to work well for the visual itself.
    """
    x = [x for x,y in points]
    y = [y for x,y in points]
    plt.subplots_adjust(right=right_shift)
    figure = plt.gcf()
    figure.set_size_inches(w=width,h=height)

    prev = 0
    for course in courses:
        print(f'prev:{prev}')
        plt.scatter(x[prev:prev+questions_per_course[course]],
                    y[prev:prev+questions_per_course[course]],
                    c = image_labels[course][0],
                    label = course,
                    marker = image_labels[course][1])
        if question_labels:
            for num in range(questions_per_course[course]):
                print(f'{course}: {num}')
                plt.annotate(num+1, (x[prev+num], y[prev+num]), fontsize=label_font)
        prev += questions_per_course[course]
    plt.legend(bbox_to_anchor=legend_loc)
    plt.savefig(image_loc, dpi=dpi)
    if show:
        plt.show()

if __name__ == "__main__":
    # courses = ['2.050J', '6.003', '8.282', '12.007', '18.600', '18.781']
    courses = ['2.016', '2.050J', '2.110J',
               '2.611', '3.012', '3.016',
               '3.091', '5.111', '6.003',
               '6.036', '6.041', '8.04',
               '8.282', '12.007', '14.01',
               '16.01+2', '16.03+4', '18.600',
               '18.781', '20.106J', 'IDS.013J',
               'Brown MATH 0180', 'Cornell CS 4420', 'Harvard STAT 110',
               'Princeton MATH 104', 'UPenn MATH 110', 'Yale PHYS 200']
    questions_per_course = {'2.016':21, '2.050J':24, '2.110J':25,
                            '2.611':27, '3.012':25, '3.016':25,
                            '3.091':25, '5.111':25, '6.003':30,
                            '6.036':30, '6.041':30, '8.04':23,
                            '8.282':20, '12.007':25, '14.01':29,
                            '16.01+2':25, '16.03+4':21, '18.600':30,
                            '18.781':20, '20.106J':25, 'IDS.013J':23,
                            'Brown MATH 0180':25, 'Cornell CS 4420':20, 'Harvard STAT 110':20,
                            'Princeton MATH 104':25, 'UPenn MATH 110':24, 'Yale PHYS 200':25}
    image_labels = {'2.016':'r.', '2.050J':'g.', '2.110J':'b.',
                    '2.611':'m.', '3.012':'k.', '3.016':'c.',
                    '3.091':'y.', '5.111':'rx', '6.003':'gx',
                    '6.036':'bx', '6.041':'mx', '8.04':'kx',
                    '8.282':'cx', '12.007':'yx', '14.01':'r+',
                    '16.01+2':'g+', '16.03+4':'b+', '18.600':'m+',
                    '18.781':'k+', '20.106J':'c+', 'IDS.013J':'y+',
                    'Brown MATH 0180':'rD', 'Cornell CS 4420':'gD', 'Harvard STAT 110':'bD',
                    'Princeton MATH 104':'mD', 'UPenn MATH 110':'kD', 'Yale PHYS 200':'cD'}
    # courses = ['18.01', '18.02', '18.03', '6.042', '18.05', '18.06', 'COMS3251']
    # questions_per_course = {course:25 for course in courses}
    # image_labels = {'18.01':'r.', '18.02':'g.', '18.03':'b.', '18.05':'mx', '18.06':'k+', '6.042':'cx', 'COMS3251':'y+'}
    for course in courses:
    #    if not os.path.exists('./embeddings/' + course + ' embeddings.json'):
    #        make_embeddings(embedding_engine, courses)
        print(len(get_embeddings('./embeddings/' + course + ' embeddings.json')))
        if len(get_embeddings('./embeddings/' + course + ' embeddings.json')) != questions_per_course[course]:
            raise Exception(f"expected {questions_per_course[course]} q's, got {len(get_embeddings('./embeddings/' + course + ' embeddings.json'))} q's in {course}")
   
    all_embeddings = []
    for course in courses:
        course_embeddings = get_embeddings('./embeddings/' + course + ' embeddings.json')
        for q_embedding in course_embeddings:
            all_embeddings.append(q_embedding)

    points_to_plot = reduce_via_umap(all_embeddings)
    plot_clusters(points_to_plot, image_location, question_labels=True, show=True)