import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def run(data_path, size):

    columns = ['size',
                'accuracy',
                'nodes',
                'leaf_nodes',
                'correction_too_cold',
                'correction_too_warm',]
    
    final_return = [size]

    '''
    learn decision tree
    '''

    from sklearn.tree import plot_tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load data
    data = pd.read_csv(data_path+'_policy.csv')

    # only take size number of samples
    data = data.iloc[:size]

    '''
    X = data.drop(columns=['action'])
    y = data['action']
    '''
    from sklearn.tree import export_graphviz

    X = data.drop(columns=['action'])
    y = data['action']

    def run_depth_experiment(max_D=50):
        depth = []
        accuracy = []
        for i in range(1, max_D+1):
            # Train decision tree
            clf = DecisionTreeClassifier(max_depth=i)
            clf.fit(X, y)

            # Test decision tree
            y_pred = clf.predict(X)
            acc = accuracy_score(y, y_pred)
            depth.append(i)
            accuracy.append(acc)
            print('Depth: ', i, 'Accuracy: ', acc)
        plt.plot(depth, accuracy)
        font_size = 20
        plt.xlabel('decision tree max depth', fontsize=font_size)
        plt.ylabel('accuracy', fontsize=font_size)
        # x tick font size
        plt.xticks(fontsize=font_size)
        # y tick font size
        plt.yticks(fontsize=font_size)
        plt.xlim(1, max_D)
        plt.grid()
        plt.tight_layout()
        plt.savefig('depth_vs_accuracy.png')
        plt.show()

    run_depth_experiment()

    # Train decision tree
    # clf = DecisionTreeClassifier(max_depth=50)
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # Test decision tree
    y_pred = clf.predict(X)
    print('Accuracy: ', accuracy_score(y, y_pred))

    final_return.append(accuracy_score(y, y_pred))

    # Plot decision tree
    # plt.figure(figsize=(50, 25))
    # plot_tree(clf, filled=True)
    # plt.savefig('decision_tree.png')
    # plt.show()

    # print the number of nodes 
    print('Number of nodes: ', clf.tree_.node_count)

    final_return.append(clf.tree_.node_count)

    # get all leaf nodes
    leaf_nodes = clf.apply(X)
    print('Leaf nodes: ', leaf_nodes)

    final_return.append(len(set(leaf_nodes)))

    # print the decision path to the first leaf node
    path = clf.decision_path(X)
    print('Path: ', path[0])

    # print the rules along this path
    rules = []
    for i in range(len(path[0].indices)-1):
        feature = clf.tree_.feature[path[0].indices[i]]
        threshold = clf.tree_.threshold[path[0].indices[i]]
        if X.columns[feature] == 'action':
            break
        if X.columns[feature] == 'state':
            rules.append('state == '+str(threshold))
        else:
            rules.append('state['+str(X.columns[feature])+'] <= '+str(threshold))

    print('Rules: ', rules)

    tree = clf

    # save the decision tree using pickle
    import pickle
    with open('temp_tree.pkl', 'wb') as f:
        pickle.dump(clf, f)


    '''
    decision path verification
    '''

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier, export_graphviz

    data = pd.read_csv(data_path+'_policy.csv')


    X = data.drop(columns=['action'])
    y = data['action']


    INPUT_VARS = ['time',
                'Site Outdoor Air Drybulb Temperature(Environment)',
                'Site Outdoor Air Relative Humidity(Environment)',
                'Site Wind Speed(Environment)',
                'Site Direct Solar Radiation Rate per Area(Environment)',
                'Zone People Occupant Count(SPACE1-1)',
                'Zone Air Temperature(SPACE1-1)']


    def decision_path_verify(tree, X_data):
        nodes_require_correction = []

        path = tree.decision_path(X_data)

        # find unique paths
        unique_path = []
        for i in range(path.shape[0]):
            p = np.where(path[i].toarray()[0] == 1)[0].tolist()
            if p not in unique_path:
                unique_path.append(p)

        # iterate through unique paths
        for i in range(len(unique_path)):
            p = unique_path[i]
            # print the rules along this path
            rules = {}
            for var in INPUT_VARS:
                rules[var] = [-99999, 99999]
            for j in range(len(p)-1):
                feature = tree.tree_.feature[p[j]]
                threshold = tree.tree_.threshold[p[j]]
                if p[j+1] == tree.tree_.children_left[j]:
                    if rules[X.columns[feature]][1] > threshold:
                        rules[X.columns[feature]][1] = threshold
                else:
                    if rules[X.columns[feature]][0] < threshold:
                        rules[X.columns[feature]][0] = threshold
            print('path #'+str(i)+' Rules: ', rules, '\nAction: ', tree.classes_[np.argmax(tree.tree_.value[p[-1]])])
            # check if the path need to be verified
            if rules['Zone People Occupant Count(SPACE1-1)'][1] > 0:
                if rules['Zone Air Temperature(SPACE1-1)'][1] > 23.5 and rules['Zone Air Temperature(SPACE1-1)'][0] >= 20:
                    # check if the path is verified
                    action = tree.classes_[np.argmax(tree.tree_.value[p[-1]])]
                    if action < rules['Zone Air Temperature(SPACE1-1)'][0]:
                        continue
                    else:
                        # correct this leaf node
                        nodes_require_correction.append([p[-1], rules, 'setpoint too warm'])
                elif rules['Zone Air Temperature(SPACE1-1)'][1] <= 23.5 and rules['Zone Air Temperature(SPACE1-1)'][0] < 20:
                    # check if the path is verified
                    action = tree.classes_[np.argmax(tree.tree_.value[p[-1]])]
                    if action > rules['Zone Air Temperature(SPACE1-1)'][1]:
                        continue
                    else:
                        # correct this leaf node
                        nodes_require_correction.append([p[-1], rules, 'setpoint too cold'])
                else:
                    continue
            else:
                continue

        return nodes_require_correction, True

    correction, verified = decision_path_verify(tree, X)

    print('No. too cold', len([c for c in correction if c[2] == 'setpoint too cold']))
    print('No. too warm', len([c for c in correction if c[2] == 'setpoint too warm']))

    final_return.append(len([c for c in correction if c[2] == 'setpoint too cold']))
    final_return.append(len([c for c in correction if c[2] == 'setpoint too warm']))

    print('\nCorrection: ', correction)
    print('Verified: ', verified)


    '''
    run simulation experiment
    '''

    import numpy as np
    import pandas as pd
    from datetime import datetime
    import gymnasium as gym
    import sinergym
    from sklearn.tree import DecisionTreeClassifier

    import data.data_manager as data_manager


    def get_action(x, model):
        return model.predict(x)

    def run_experiment(city, model_path, log=True, monitor_path=None, overhead_path=None, winter=True):

        new_action_mapping = {
            0: (15, 30),
            1: (16, 29),
            2: (17, 28),
            3: (18, 27),
            4: (19, 26),
            5: (20, 25),
            6: (21, 24),
            7: (22, 23),
            8: (22, 22),
            9: (21, 21)
        }

        if winter:
            extra_params={'timesteps_per_hour' : 4,
                    'runperiod' : (1,1,1997,31,1,1997)}
        else:
            extra_params={'timesteps_per_hour' : 4,
                    'runperiod' : (1,7,1997,31,7,1997)}

        if city == 'pittsburgh':
            env = gym.make('Eplus-demo-v1', config_params=extra_params, action_mapping=new_action_mapping)
        elif city == 'tucson':
            env = gym.make('Eplus-5Zone-hot-discrete-v1', config_params=extra_params, action_mapping=new_action_mapping)
        elif city == 'ny':
            env = gym.make('Eplus-5Zone-mixed-discrete-v1', config_params=extra_params, action_mapping=new_action_mapping)

        obs, info = env.reset()
        terminated = False
        time_overhead = []

        current_step = 0

        monitor = pd.DataFrame(columns=['year',
                                        'month',
                                        'day',
                                        'hour',
                                        'Site Outdoor Air Drybulb Temperature(Environment)',
                                        'Site Outdoor Air Relative Humidity(Environment)',
                                        'Site Wind Speed(Environment)',
                                        'Site Wind Direction(Environment)',
                                        'Site Diffuse Solar Radiation Rate per Area(Environment)',
                                        'Site Direct Solar Radiation Rate per Area(Environment)',
                                        'Zone Thermostat Heating Setpoint Temperature(SPACE1-1)',
                                        'Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)',
                                        'Zone Air Temperature(SPACE1-1)',
                                        'Zone Air Relative Humidity(SPACE1-1)',
                                        'Zone People Occupant Count(SPACE1-1)',
                                        'People Air Temperature(SPACE1-1 PEOPLE 1)',
                                        'Facility Total HVAC Electricity Demand Rate(Whole Building)'])

        input_vars = ['time',
                    'Site Outdoor Air Drybulb Temperature(Environment)',
                    'Site Outdoor Air Relative Humidity(Environment)',
                    'Site Wind Speed(Environment)',
                    'Site Direct Solar Radiation Rate per Area(Environment)',
                    'Zone People Occupant Count(SPACE1-1)',
                    'Zone Air Temperature(SPACE1-1)']
        
        input_var_idx = [0, 4, 5, 6, 9, 14, 12]

        # read decision tree classifier model from model_path (.pkl)
        import pickle
        pickle_in = open(model_path, "rb")

        model = pickle.loads(pickle_in.read())

        while not terminated:
            t0 = datetime.now()
            if current_step == 0:
                in_obs = [0, 0, 0, 0, 0, 0, 0]
                # reshape obs to 2D array
                in_obs = np.reshape(in_obs, (1, -1))
                # add feature names to in_obs
                in_obs = pd.DataFrame(in_obs, columns=input_vars)
            else:
                # time is current_step mod 96 times 0.25
                time = (current_step % 96) * 0.25
                # get other variables in input_vars from obs
                in_obs = [time]
                for var in input_vars[1:]:
                    in_obs.append(obs[var])
                # reshape obs to 2D array
                in_obs = np.reshape(in_obs, (1, -1))
                # add feature names to in_obs
                in_obs = pd.DataFrame(in_obs, columns=input_vars)
            action = get_action(in_obs, model)
            if winter:
                for i in new_action_mapping.keys():
                    if new_action_mapping[i][0] == int(action[0]):
                        action = i
                        break
            else:
                continue
            t1 = datetime.now()
            time_overhead.append(t1-t0)

            obs, reward, terminated, truncated, info = env.step(action)
            # add each item in the obs (a dictionary) to the monitor dataframe using concat
            monitor = pd.concat([monitor, pd.DataFrame(obs, index=[current_step])])

            current_step += 1
            if current_step % 100 == 0:
                print(current_step)
        
        # turn time overhead into integer in milliseconds
        time_overhead = [int(t.total_seconds() * 1000) for t in time_overhead]
        # save time overhead to csv
        time_overhead_df = pd.DataFrame(time_overhead, columns=['time_overhead'])
        time_overhead_df.to_csv(overhead_path)
        # save monitor to csv 
        monitor.to_csv(monitor_path)
        env.close()


    run_experiment(city='pittsburgh', 
                    model_path='temp_tree.pkl', 
                    monitor_path='zresults/pittsburgh_tree_monitor_{}.csv'.format(size), 
                    overhead_path='zresults/pittsburgh_tree_overhead_{}.csv'.format(size), 
                    winter=True)
    
    return final_return

# Load data
data_path = 'IP_decisions_2/IP_decisions_noise=0.01_model=1200'
test_data = []
for size in range(10, 1000, 50):
    test_data.append(run(data_path, size))
    test_data_csv = pd.DataFrame(test_data, columns=['size',
                                                    'accuracy',
                                                    'nodes',
                                                    'leaf_nodes',
                                                    'correction_too_cold',
                                                    'correction_too_warm'])
    test_data_csv.to_csv('test_data_pittsburgh.csv')
