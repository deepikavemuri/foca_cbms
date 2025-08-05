import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet101,
    resnet50,
    ResNet101_Weights,
    resnet18,
    ResNet18_Weights,
    ResNet50_Weights,
)


class FoCA_CBM(nn.Module):
    def __init__(
        self,
        intent_list,
        fc_list,
        backbone_layer_ids,
        num_classes,
        pretrained_clfs_path=None,
        pretrained_attrs_path=None,
        pretrained_backbone_path=None,
        backbone_name="resnet101",
        exclusive_attrs=False,
    ):
        super(FoCA_CBM, self).__init__()

        # print("Intent list: ", [len(q) for q in intent_list])
        # print("FC list: ", [len(q) for q in fc_list])
        self.intent_list = intent_list
        self.fc_list = fc_list
        self.exclusive_attrs = exclusive_attrs
        self.backbone_layer_ids = backbone_layer_ids
        if backbone_name == "resnet18":
            int_output_sizes = [64, 128, 256, 512]  # Resnet18 specific sizes
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone_name == "resnet50":
            int_output_sizes = [256, 512, 1024, 2048]  # Resnet50 specific sizes
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone_name == "resnet101":
            int_output_sizes = [256, 512, 1024, 2048]  # Resnet specific sizes
            backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        # backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.layers = nn.ModuleDict()
        self.layers["1"] = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,  # ReLU activation if using timm.create_model if using pytorch resnet101, put relu
            backbone.maxpool,
            backbone.layer1,
        )
        self.layers["2"] = backbone.layer2
        self.layers["3"] = backbone.layer3
        self.layers["4"] = backbone.layer4

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dynamically creating the attribute layers and classifiers
        self.attr_layers = nn.ModuleList()
        self.attr_bn_layers = nn.ModuleList()
        self.classifier_layers = nn.ModuleList()

        for i, intent in enumerate(intent_list):
            layer_id = backbone_layer_ids[i] - 1  # Adjust for 0-indexing

            # Create the attribute layer
            attr_layer = nn.Linear(int_output_sizes[layer_id], len(intent))
            self.attr_layers.append(attr_layer)
            self.attr_bn_layers.append(nn.BatchNorm1d(len(intent)))

            # Create the classifier layer
            if self.exclusive_attrs:
                clf_size = sum(
                    len(i) for i in intent_list[: i + 1]
                )  # All attributes till now
            else:
                clf_size = len(intent)  # Only the current attributes
            clf_layer = nn.Linear(clf_size, num_classes)
            # parametrize.register_parametrization(clf_layer, "weight", nn.ReLU())      # Enforcing the classifier weights to be positive.
            self.classifier_layers.append(clf_layer)

        # Load pretrained attr_layers and freeze
        if pretrained_attrs_path:
            weights = torch.load(pretrained_attrs_path)
            for i, attr_layer in enumerate(self.attr_layers):
                attr_layer.load_state_dict(
                    dict(
                        {
                            "weight": weights[f"attr_layers.{i}.weight"],
                            "bias": weights[f"attr_layers.{i}.bias"],
                        }
                    )
                )

            for attr_layer in self.attr_layers:
                for param in attr_layer.parameters():
                    param.requires_grad = False

        # Load pretrained classifiers and freeze
        if pretrained_clfs_path:
            weights = torch.load(pretrained_clfs_path)
            for i, clf in enumerate(self.classifier_layers):
                clf.load_state_dict(
                    dict(
                        {
                            "weight": weights[f"classifier_layers.{i}.weight"],
                            "bias": weights[f"classifier_layers.{i}.bias"],
                        }
                    )
                )

            # for clf in self.classifier_layers:
            #     for param in clf.parameters():
            #         param.requires_grad = False

        # Load pretrained backbone and freeze
        if pretrained_backbone_path:
            weights = torch.load(pretrained_backbone_path)
            backbone_state_dict = {
                k: v
                for k, v in weights.items()
                if not (k.startswith("attr") or k.startswith("classifier"))
            }
            self.layers.load_state_dict(backbone_state_dict, strict=False)
            for layer in self.layers.values():
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        attr_non_sig_preds, attr_preds, class_preds = [], [], []
        attr_counter = 0

        for i in self.layers.keys():
            # Forward through the backbone layers
            x = self.layers[i](x)
            if int(i) in self.backbone_layer_ids:
                attr_layer = self.attr_layers[attr_counter]
                attr_bn_layer = self.attr_bn_layers[attr_counter]
                clf_layer = self.classifier_layers[attr_counter]

                gap_output = self.global_avg_pool(x)
                attr_pred = attr_layer(gap_output.squeeze(-1).squeeze(-1))
                attr_pred = attr_bn_layer(attr_pred)
                attr_non_sig_preds.append(attr_pred)
                attr_preds.append(F.sigmoid(attr_pred))

                if self.exclusive_attrs:
                    attrs_till_now = torch.cat(
                        attr_non_sig_preds[: attr_counter + 1], dim=1
                    )
                    class_pred = clf_layer(attrs_till_now)
                else:
                    class_pred = clf_layer(F.sigmoid(attr_pred))

                # Apply softmax for the final class predictions
                class_preds.append(class_pred)
                attr_counter += 1

        for i in range(len(class_preds)):
            if i > 0:
                class_preds[i] *= F.sigmoid(class_preds[i - 1])

        return attr_preds, class_preds

    def clf_weight_initialization(self, concept_set, cls_list):
        for i in range(len(self.classifier_layers)):
            w = torch.zeros_like(self.classifier_layers[i].weight)
            b = torch.zeros_like(self.classifier_layers[i].bias)

            # print(concept_set.keys())
            j = 0
            for cls in concept_set.keys():
                if cls in cls_list:
                    for k, intent in enumerate(self.intent_list[i]):
                        if intent in concept_set[cls]:
                            w[j, k] = 1
                    j += 1
            self.classifier_layers[i].weight.data = w
            self.classifier_layers[i].bias.data = b

    def l1_regularize(self, lambda_l1=1e-3):
        l1_loss = 0
        # Add L1 loss for each attribute layer
        for attr_layer in self.attr_layers:
            for param in attr_layer.parameters():
                l1_loss += torch.sum(torch.abs(param))
        return lambda_l1 * l1_loss

    def clf_weight_l1_regularize(self, lambda_l1=1e-3):
        l1_loss = 0
        # Add L1 loss for each attribute layer
        for clf_layer in self.classifier_layers:
            for n, param in clf_layer.named_parameters():
                if "weight" in n:
                    l1_loss += torch.sum(torch.abs(param))
        return lambda_l1 * l1_loss

    # elasticnet weight regularization on classifier weights
    def clf_weight_elasticnet_regularize(self, lam=1e-3, alpha=1e-3):
        l1_loss = 0
        l2_loss = 0
        # Add L1 loss for each attribute layer
        for clf_layer in self.classifier_layers:
            for n, param in clf_layer.named_parameters():
                if "weight" in n:
                    l1_loss += param.norm(p=1)
                    l2_loss += torch.sum(param**2)
        return lam * alpha * l1_loss + 0.5 * lam * (1 - alpha) * l2_loss


class CBM(nn.Module):
    def __init__(self, model_name, num_classes, num_attrs):
        super(CBM, self).__init__()
        if model_name == "resnet18":
            out_size = 512
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == "resnet50":
            out_size = 2048
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Invalid model name")
        self.model.fc = nn.Identity()
        self.bottleneck = nn.Sequential(nn.Linear(out_size, num_attrs))
        self.classifier = nn.Linear(num_attrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        concepts = self.bottleneck(x)
        classes = self.classifier(concepts)
        return concepts, classes


class FoCA_CBM_N(nn.Module):
    def __init__(
        self,
        intent_list,
        fc_list,
        num_classes,
        backbone_name="resnet101",
    ):
        super(FoCA_CBM_N, self).__init__()

        self.intent_list = intent_list
        self.fc_list = fc_list
        if backbone_name == "resnet18":
            out_size = 512
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone_name == "resnet50":
            out_size = 2048
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone_name == "resnet101":
            out_size = 2048
            self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Invalid model name")

        self.model.fc = nn.Identity()
        self.attr1 = nn.Sequential(nn.Linear(out_size, len(intent_list[0])))
        self.attr2 = nn.Sequential(nn.Linear(len(intent_list[0]), len(intent_list[1])))
        self.classifier = nn.Linear(len(intent_list[1]), num_classes)

    def forward(self, x):
        x = self.model(x)
        attr1_pred = self.attr1(x)  # F.sigmoid(self.attr1(x))
        attr2_pred = self.attr2(attr1_pred)  # F.sigmoid(self.attr2(attr1_pred))
        classes = self.classifier(attr2_pred)
        return attr1_pred, attr2_pred, classes


class MLPCBM(nn.Module):
    def __init__(self, model_name, num_classes, num_attrs, expand_dim=None):
        super(MLPCBM, self).__init__()
        self.expand_dim = expand_dim
        if model_name == "resnet18":
            out_size = 512
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == "resnet50":
            out_size = 2048
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Invalid model name")
        self.model.fc = nn.Identity()
        if expand_dim:
            self.bottleneck = nn.Sequential(nn.Linear(out_size, expand_dim))
            self.activation = torch.nn.ReLU()
            self.linear = nn.Linear(expand_dim, num_attrs)
        else:
            self.bottleneck = nn.Linear(out_size, num_attrs)
        self.classifier = nn.Linear(num_attrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        concepts = self.bottleneck(x)
        if self.expand_dim:
            concepts = self.activation(concepts)
            concepts = self.linear(concepts)
        classes = self.classifier(concepts)
        return concepts, classes
