import json

import scrapy
from scrapy.crawler import CrawlerProcess

data = []


class BankiSpider(scrapy.Spider):
    name = 'BankiSpider'

    def __init__(self, domain='www.banki.ru/services/responses/bank/tcs', pages_count=500):
        self.domain = domain
        self.pages_count = int(pages_count)
        self.data = []

    def start_requests(self):
        for page in range(1, self.pages_count + 1):
            print('http://{}/?page={}'.format(self.domain, page))
            yield scrapy.Request('http://{}/?page={}'.format(self.domain, page))

    @staticmethod
    def extract_text(text):
        return '\n'.join(map(lambda it: it.strip(), text.css('::text').extract()))

    @staticmethod
    def extract_rating(rating):
        fields = rating.css('span.rating-grade ::text').extract()
        return int(fields[1].strip()) if len(fields) == 2 else None

    def parse(self, response):
        feedbacks = response.css('article.responses__item')

        for feedback in feedbacks:
            message = feedback.css('div.responses__item__message')
            rating = feedback.css('div.responses__item__rating')

            entry = {
                'text': self.extract_text(message),
                'rating': self.extract_rating(rating)
            }

            global data
            data.append(entry)

            yield entry


if __name__ == '__main__':
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    process.crawl(BankiSpider)
    process.start()

    with open("data.json", "w") as f:
        json.dump(data, f)
