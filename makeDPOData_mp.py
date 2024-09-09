from src.module.readData import read_all_data






"conversations": [
      {
        "from": "human",
        "value": "Please provide a review based on the following paper, including a summary, strengths, weaknesses, and any questions you have: Title: [title]. Abstract: [abstract]. Main Text: [paper]."
      }
    ],
    "chosen": {   # 对于接收论文，这里用正分review；对于被拒论文，这里用负分review， 这里是一致的正常review
      "from": "gpt",
      "value": "Summary: [summary]. Strengths: [strengths]. Weaknesses: [weakness]. Questions: [questions]."
    },
    "rejected": { # 对于接收论文，这里用负分review；对于被拒论文，这里用正分review， 这里是不一致的傻逼review
      "from": "gpt",
      "value": "Summary: [summary]. Strengths: [strengths]. Weaknesses: [weakness]. Questions: [questions]."
    }
  



def ICLR_formatter():
    result = {
        "conversations" = [],
        "chosen" = {},
        "rejected" = {}
    }


def main():
    _, _, iclr_review_path, iclr_paper_path, _, _ = read_all_data()


