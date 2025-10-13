# Copyright 2016 Symantec Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

import bisect
import collections
import itertools
import math
import operator
import os
import random
import shelve

import numpy

import model
import ngram_chain
import DP_util

TmpNode = collections.namedtuple('TmpNode',
                                 'transitions probabilities '
                                 'cumprobs logprobs')


class DP_BackoffModel(ngram_chain.NGramModel):

    def __init__(self, words, threshold, start_symbol=True,
                 with_counts=False, epsilon_total=1.0, delta_total=0.05, shelfname=None):

        if not with_counts:
            words = [(1, w) for w in words]

        self.start = start = '\0' if start_symbol else ''
        self.end = end = '\0'
        lendelta = len(start) + len(end)
        words = [(len(w) + lendelta, (c, start + w + end))
                 for c, w in words]

        self.nodes = nodes = self.setup_nodes(shelfname)

        words.sort(key=operator.itemgetter(0))
        if not words:
            return

        lens, words = zip(*words)
        lens = numpy.array(lens)
        U = lens.max()
        nwords = len(words)

        def zerodict():
            return collections.defaultdict(itertools.repeat(0).__next__)

        charcounts = zerodict()
        for count, word in words:
            for c in word[start_symbol:]:
                charcounts[c] += count

        epsilon = epsilon_total / 10
        for i, char in enumerate(charcounts):
            charcounts[char] = max(1, charcounts[char] + numpy.random.laplace(0, 2 * (U-lendelta) / epsilon, size=1).item())
        totchars = sum(charcounts.values())
        transitions, counts = zip(*sorted(charcounts.items(),
                                          key=operator.itemgetter(1),
                                          reverse=True))
        transitions = ''.join(transitions)
        counts = numpy.array(counts)
        totchars = counts.sum()

        probabilities = counts / totchars

        if transitions:
            nodes[''] = TmpNode(transitions,
                                probabilities,
                                probabilities.cumsum(),
                                -numpy.log2(probabilities))

        leftidx = 0
        skipwords = set()

        # for n in range(2, lens[-threshold] + 1):
        for n in range(2, 11):

            leftidx = bisect.bisect_left(lens, n)

            ngram_counter = zerodict()
            for i in range(leftidx, nwords):
                if i in skipwords:
                    continue
                count, word = words[i]
                skip = True
                for j in range(lens[i] - n + 1):
                    ngram = word[j: j + n]
                    if ngram[:-2] in nodes:
                        ngram_counter[ngram] += count
                        skip = False
                if skip:
                    skipwords.add(i)

            tmp_dict = collections.defaultdict(list)
            for ngram, count in ngram_counter.items():
                tmp_dict[ngram[:-1]].append((ngram[-1], count))

            threshold_DP = threshold + numpy.random.laplace(0, 2*2*(U+2-n) / (epsilon/2), size=1).item()

            noise_scale = 2 * (U + 2 - n) / (epsilon/2)

            for state, sscounts in tmp_dict.items():
                total = sum(count for _, count in sscounts)
                if total < threshold:
                    continue
                sscounts_with_noise = {}
                for char, count in sscounts:
                    laplace_noise = numpy.random.laplace(0, noise_scale, size=1)
                    noisy_count = max(threshold, count + laplace_noise[0])
                    sscounts_with_noise[char] = noisy_count
                total_noise = sum(sscounts_with_noise.values())
                trans_probs = {c: sscounts_with_noise[c] / total_noise
                               for c, count in sscounts
                               if DP_util.AboveThreshold(count, threshold_DP, epsilon=(epsilon/2), delta_u = 2*(U+2-n))}
                missing = 1 - sum(trans_probs.values())
                if missing == 1:
                    continue

                if missing > 0:
                    i = 1
                    parent_state = None
                    while parent_state is None:
                        parent_state = self.nodes.get(state[i:], None)
                        if parent_state is None:
                            i += 1
                        if i > len(state):
                            parent_state = self.nodes.get('', None)
                    for c, p in zip(parent_state.transitions, parent_state.probabilities):
                        trans_probs[c] = trans_probs.get(c, 0) + p * missing
                trans_probs = sorted(trans_probs.items(),
                                     key=operator.itemgetter(1),
                                     reverse=True)
                transitions, probabilities = zip(*trans_probs)
                transitions = ''.join(transitions)
                probabilities = numpy.array(probabilities)
                # probabilities must sum to 1
#                assert abs(probabilities.sum() - 1) < 0.001
                nodes[state] = TmpNode(transitions, probabilities,
                                       probabilities.cumsum(),
                                       -numpy.log2(probabilities))
         
        Node = ngram_chain.Node
        for state, node in self.nodes.items():
            nodes[state] = Node(node.transitions, node.cumprobs,
                                node.logprobs)

    def update_state(self, state, transition):
        nodes = self.nodes
        state += transition
        while state not in nodes:
            state = state[1:]
        return state


class LazyBackoff(model.Model):

    def __init__(self, path, threshold, start=True, end=True):
        self.threshold = threshold
        self.start = '\0' if start else ''
        self.end = '\0' if end else ''
        self.shelves = {
            int(fname): shelve.open(os.path.join(path, fname), 'r')
            for fname in os.listdir(path)
        }

    def hasnode(self, state):
        shelves = self.shelves
        return len(state) in shelves and state in shelves[len(state)]

    def getnode(self, state):
        return self.shelves[len(state)][state]

    def getcount(self, ngram):
        if ngram != '':
            return self.shelves[len(ngram) - 1][ngram[:-1]][ngram[-1]]
        else:
            return sum(self.shelves[0][''].values())

    def begin(self):
        start = self.start
        return start, self.getnode(start)

    def update_state(self, state, transition):
        state += transition
        while not self.hasnode(state):
            state = state[1:]
        while self.getcount(state) < self.threshold:
            state = state[1:]
        node = self.getnode(state)
        return state, node

    def backoff(self, state):
        state = state[1:]
        node = self.getnode(state)
        return state, node

    def logprob(self, word, leaveout=False):

        res = 0
        state, node = self.begin()
        for c in word + self.end:
            while True:  # break when we should stop backing off
                count = node.get(c, 0) - leaveout
                total = self.getcount(state) - leaveout
                if state == self.start == self.end:
                    total /= 2
                if count >= self.threshold or state == '':
                    break
                passing = sum(ct for ct in node.values()
                              if ct >= self.threshold)
                if leaveout and count == self.threshold - 1:
                    passing -= self.threshold
                try:
                    res -= math.log2(1 - (passing / total))
                except ValueError:
                    return float('inf')
                state, node = self.backoff(state)
            try:
                res -= math.log2(count / total)
            except ValueError:
                return float('inf')
            state, node = self.update_state(state, c)
        return res

    def generate(self, maxlen=100):

        logprob = 0
        state, node = self.begin()
        word = ''
        while True:  # return when we find self.end
            while True:  # break when we should stop backing off
                values = list(node.values())
                cumsum = numpy.cumsum(values)
                total = self.getcount(state)
                if state == self.start == self.end:
                    total /= 2
                idx = bisect.bisect_right(cumsum, random.randrange(total))
                if idx < len(values):
                    count = values[idx]
                    if state == '' or count >= self.threshold:
                        break
                state, node = self.backoff(state)
                passing = sum(ct for ct in values if ct >= self.threshold)
                logprob -= math.log2(1 - (passing / total))
            logprob -= math.log2(count / total)
            c = next(itertools.islice(node.keys(), idx, None))  # n-th elem
            if c == self.end:
                return logprob, word
            word += c
            if len(word) >= maxlen:
                return logprob, word
            state, node = self.update_state(state, c)
