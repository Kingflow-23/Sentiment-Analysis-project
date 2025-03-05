# --------------------------------------------------------------------------   
# Manually expanded dataset with equal distribution of content and labels (15 entries per label)
FAKE_DATASET = {
    "content": [
        # Really Positive (label 5)
        "I love this app! 😍",  
        "This is exactly what I needed, I'm so happy with it!",  
        "Such a great experience, I would definitely recommend this!",  
        "Amazing quality, worth every penny!",  
        "I can’t stop using this, it’s so good!",  
        "Best decision I’ve ever made, I’m thrilled!",  
        "Exceeded my expectations in every way! Highly recommend!",  
        "I am absolutely in love with this product, it’s perfect!",  
        "Incredible, I will buy it again for sure!",  
        "I’m extremely satisfied, it’s better than I imagined.",  
        "Fantastic quality, I will be a lifelong customer.",  
        "I’m so happy with this, it changed my life!",  
        "I’ve never been this satisfied with a product before.",  
        "Such a great investment, totally worth it!",  
        "Beyond perfect, I can’t imagine living without it!",  

        # Positive (label 4)
        "Really enjoying this, it's made my life easier.",  
        "Great app, everything works as expected!",  
        "This is fantastic, I recommend it to all my friends!",  
        "I’m very happy with the quality of the product.",  
        "Good value for money, it does exactly what it promises.",  
        "I would definitely buy this again!",  
        "One of the best purchases I’ve made!",  
        "I feel so much more organized with this app.",  
        "It’s a great product, really glad I got it!",  
        "I’m satisfied with my purchase.",  
        "Great service, no complaints!",  
        "This product works perfectly and I’m happy with it.",  
        "I’m glad I bought this, works as expected.",  
        "This product is very useful, I'm happy with it.",  
        "Good quality and easy to use.",  

        # Neutral (label 3)
        "It works fine, but nothing extraordinary.",  
        "The product is okay, just as described.",  
        "I didn’t have any major issues with it, but it’s not perfect.",  
        "It’s decent, I guess, but I’ve seen better.",  
        "The product is alright, but I wouldn’t recommend it to others.",  
        "It’s neither good nor bad, just average.",  
        "Meh, it does the job but nothing exciting.",  
        "It’s okay, I don’t really have strong feelings about it.",  
        "It works, but it’s not the best option out there.",  
        "Just average, nothing special about it.",  
        "The product is fine, but it could use some improvements.",  
        "It’s not the best, but it gets the job done.",  
        "It’s a good product, just not great.",  
        "Nothing to complain about, but nothing to praise either.",  
        "It’s functional, but I expected more for the price.",  

        # Negative (label 2)
        "The product stopped working after a week, very disappointing.",  
        "Terrible customer service, I’m not coming back.",  
        "The quality is awful, I regret buying it.",  
        "It’s way too slow and not worth the price.",  
        "Not what I expected, very dissatisfied.",  
        "It broke after a few uses. Very unhappy.",  
        "The material feels cheap and flimsy, wouldn’t buy again.",  
        "Waste of money, completely useless.",  
        "I am so disappointed with my purchase.",  
        "Not worth it, I wouldn’t recommend this to anyone.",  
        "The customer service was terrible, never again.",  
        "It doesn’t work as advertised, so frustrating.",  
        "I should have read the reviews before purchasing.",  
        "I regret this purchase, I expected much more.",  
        "Very poor quality, won’t be buying again.",  

        # Really Negative (label 1)
        "Completely awful, I hate it.",  
        "This was a nightmare, never buying from here again.",  
        "Total disappointment, the worst purchase ever!",  
        "Worst purchase in my life, do not buy it!",  
        "This is the most frustrating product I’ve ever used.",  
        "It’s terrible. Waste of money.",  
        "Don’t even think about buying this, it’s awful.",  
        "Regret purchasing this. Worst experience ever.",  
        "This product ruined my day, don’t waste your time or money.",  
        "I’ll never buy from this brand again, horrible quality.",  
        "I absolutely hate this. Worst decision ever.",  
        "This is garbage, totally unusable.",  
        "It broke within an hour of use, I’m furious!",  
        "The worst item I’ve ever received. So disappointed.",  
        "I’m returning this immediately. It’s just awful.",  
    ],

    "score": [
        # Really Positive (label 5)
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,

        # Positive (label 4)
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,

        # Neutral (label 3)
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,

        # Negative (label 2)
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

        # Really Negative (label 1)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ]
}