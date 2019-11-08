from scripts import downloadData
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument(
    "--downloadLocation",
    type=str,
    default=os.path.join("data", "cytodata_2019_orig_challenge_data.zip"),
    dest="downloadLocation",
    help="The account name of the blobStorage where the validation data is stored",
)
parser.add_argument(
    "--downloadUrl",
    type=str,
    default=r"https://ndownloader.figshare.com/files/18501824?private_link=f41918598b1ff5116825",
    dest="downloadUrl",
    help="The account name of the blobStorage where the validation data is stored",
)

args = parser.parse_args()

downloadData(args.downloadLocation, args.downloadUrl)
