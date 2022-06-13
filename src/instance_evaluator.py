import shapely.wkt
import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm
import os
from solaris.eval import iou
from fiona.errors import DriverError
from fiona._err import CPLE_OpenFailedError


class Evaluator():
    """Object to test IoU for predictions and ground truth polygons.

    Attributes
    ----------
    ground_truth_fname : str
        The filename for the ground truth CSV or JSON.
    ground_truth_GDF : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` containing the ground truth vector
        labels.
    ground_truth_GDF_Edit : :class:`geopandas.GeoDataFrame`
        A copy of ``ground_truth_GDF`` which will be manipulated during
        processing.
    proposal_GDF : :class:`geopandas.GeoDataFrame`
        The proposal :class:`geopandas.GeoDataFrame`, added using
        ``load_proposal()``.

    Arguments
    ---------
    ground_truth_vector_file : str
        Path to .geojson file for ground truth.

    """

    def __init__(self, ground_truth_vector_file, proposal_gdf):
        # Load Ground Truth : Ground Truth should be in geojson or shape file
        try:
            if ground_truth_vector_file.lower().endswith('json'):
                self.load_truth(ground_truth_vector_file)
            elif ground_truth_vector_file.lower().endswith('csv'):
                self.load_truth(ground_truth_vector_file, truthCSV=True)
            self.ground_truth_fname = ground_truth_vector_file
        except AttributeError:  # handles passing gdf instead of path to file
            self.ground_truth_GDF = ground_truth_vector_file
            self.ground_truth_fname = 'GeoDataFrame variable'
        self.ground_truth_sindex = self.ground_truth_GDF.sindex  # get sindex
        # create deep copy of ground truth file for calculations
        self.ground_truth_GDF_Edit = self.ground_truth_GDF.copy(deep=True)
        self.proposal_GDF = proposal_gdf
        self.proposal_GDF['__total_conf'] = 1.0
        self.proposal_GDF['__max_conf_class'] = 1

    def __repr__(self):
        return 'Evaluator {}'.format(os.path.split(
            self.ground_truth_fname)[-1])

    def get_iou_by_building(self):
        """Returns a copy of the ground truth table, which includes a
        per-building IoU score column after eval_iou_spacenet_csv() has run.
        """

        output_ground_truth_GDF = self.ground_truth_GDF.copy(deep=True)
        return output_ground_truth_GDF


    def eval_iou(self, miniou=0.5, iou_field_prefix='iou_score',
                 ground_truth_class_field='', calculate_class_scores=True,
                 class_list=['all']):
        """Evaluate IoU between the ground truth and proposals.

        Arguments
        ---------
        miniou : float, optional
            Minimum intersection over union score to qualify as a successful
            object detection event. Defaults to ``0.5``.
        iou_field_prefix : str, optional
            The name of the IoU score column in ``self.proposal_GDF``. Defaults
            to ``"iou_score"``.
        ground_truth_class_field : str, optional
            The column in ``self.ground_truth_GDF`` that indicates the class of
            each polygon. Required if using ``calculate_class_scores``.
        calculate_class_scores : bool, optional
            Should class-by-class scores be calculated? Defaults to ``True``.
        class_list : list, optional
            List of classes to be scored. Defaults to ``['all']`` (score all
            classes).

        Returns
        -------
        scoring_dict_list : list
            list of score output dicts for each image in the ground
            truth and evaluated image datasets. The dicts contain
            the following keys: ::

                ('class_id', 'iou_field', 'TruePos', 'FalsePos', 'FalseNeg',
                'Precision', 'Recall', 'F1Score')

        """

        scoring_dict_list = []

        if calculate_class_scores:
            if not ground_truth_class_field:
                raise ValueError('Must provide ground_truth_class_field '
                                 'if using calculate_class_scores.')
            if class_list == ['all']:
                class_list = list(
                    self.ground_truth_GDF[ground_truth_class_field].unique())
                if not self.proposal_GDF.empty:
                    class_list.extend(
                        list(self.proposal_GDF['__max_conf_class'].unique()))
                class_list = list(set(class_list))

        for class_id in class_list:
            iou_field = "{}_{}".format(iou_field_prefix, class_id)
            if class_id is not 'all':  # this is probably unnecessary now
                self.ground_truth_GDF_Edit = self.ground_truth_GDF[
                    self.ground_truth_GDF[
                        ground_truth_class_field] == class_id].copy(deep=True)
            else:
                self.ground_truth_GDF_Edit = self.ground_truth_GDF.copy(
                    deep=True)

            for _, pred_row in self.proposal_GDF.iterrows():
                if pred_row['__max_conf_class'] == class_id \
                   or class_id == 'all':
                    pred_poly = pred_row.geometry
                    iou_GDF = iou.calculate_iou(pred_poly,
                                                self.ground_truth_GDF_Edit)
                    # Get max iou
                    if not iou_GDF.empty:
                        max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(
                            axis=0, skipna=True)]
                        if max_iou_row['iou_score'] > miniou:
                            self.proposal_GDF.loc[pred_row.name, iou_field] \
                                = max_iou_row['iou_score']
                            self.ground_truth_GDF_Edit \
                                = self.ground_truth_GDF_Edit.drop(
                                    max_iou_row.name, axis=0)
                        else:
                            self.proposal_GDF.loc[pred_row.name, iou_field] = 0
                    else:
                        self.proposal_GDF.loc[pred_row.name, iou_field] = 0

            if self.proposal_GDF.empty:
                TruePos = 0
                FalsePos = 0
            else:
                try:
                    TruePos = self.proposal_GDF[
                        self.proposal_GDF[iou_field] >= miniou].shape[0]
                    FalsePos = self.proposal_GDF[
                        self.proposal_GDF[iou_field] < miniou].shape[0]
                except KeyError:  # handle missing iou_field
                    print("iou field {} missing")
                    TruePos = 0
                    FalsePos = 0

            # number of remaining rows in ground_truth_gdf_edit after removing
            # matches is number of false negatives
            FalseNeg = self.ground_truth_GDF_Edit.shape[0]
            if float(TruePos+FalsePos) > 0:
                Precision = TruePos / float(TruePos + FalsePos)
            else:
                Precision = 0
            if float(TruePos + FalseNeg) > 0:
                Recall = TruePos / float(TruePos + FalseNeg)
            else:
                Recall = 0
            if Recall*Precision > 0:
                F1Score = 2*Precision*Recall/(Precision+Recall)
            else:
                F1Score = 0

            score_calc = {'class_id': class_id,
                          'iou_field': iou_field,
                          'TruePos': TruePos,
                          'FalsePos': FalsePos,
                          'FalseNeg': FalseNeg,
                          'Precision': Precision,
                          'Recall':  Recall,
                          'F1Score': F1Score
                          }
            scoring_dict_list.append(score_calc)
            
        return TruePos, FalsePos, FalseNeg