from sentimentnetwork import mlp

def main():
    pass

if __name__ == "__main__":
    reviews = ["I loved this movie", "This movie sucked!!!",\
    "It was okay...", "Baffling performances",\
    "I was at the edge of my seat"]
    # for r in reviews:
    #     sentiment = mlp.run(r)
    #     print(r)
    #     print(sentiment + "\n" * 2)
    while(1):
        review = input("Please submit a movie review!!!\n")
        sentiment = mlp.run(review)
        print("\n" + review)
        print(sentiment + "\n" * 4)
