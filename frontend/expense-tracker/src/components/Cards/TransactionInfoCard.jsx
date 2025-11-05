import React from "react";
import {
  LuUtensils,
  LuTrendingDown,
  LuTrendingUp,
  LuTrash2,
  LuPencil,
} from "react-icons/lu";

const TransactionInfoCard = ({
  title,
  icon,
  date,
  amount,
  type,
  hideDeleteBtn,
  onDelete,
  onEdit,
  notes,
}) => {
  const getAmountStyles = () =>
    type === "income" ? "bg-green-50 text-green-500" : "bg-red-50 text-red-500";

  return (
    <div className="group relative flex flex-col gap-2 mt-2 p-3 rounded-lg hover:bg-gray-100/60">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 flex items-center justify-center text-xl text-gray-800 bg-gray-100 rounded-full">
          {icon ? (
            icon.startsWith("http") ? (
              <img src={icon} alt={title} className="w-6 h-6" />
            ) : (
              <span className="text-2xl">{icon}</span>
            )
          ) : (
            <span className="text-2xl">{type === "income" ? "💵" : "💸"}</span>
          )}
        </div>
        <div className="flex-1 flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-700 font-medium">{title}</p>
            <p className="text-xs text-gray-400 mt-1">{date}</p>
          </div>
          <div className="flex items-center gap-2">
            {!hideDeleteBtn && (
              <>
                <button
                  className="text-gray-400 hover:text-blue-500 transition-colors cursor-pointer"
                  onClick={onEdit}
                  title="Edit"
                >
                  <LuPencil size={18} />
                </button>
                <button
                  className="text-gray-400 hover:text-red-500 transition-colors cursor-pointer"
                  onClick={onDelete}
                  title="Delete"
                >
                  <LuTrash2 size={18} />
                </button>
              </>
            )}
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md ${getAmountStyles()}`}
            >
              <h6 className="text-xs font-medium">
                {type === "income" ? "+" : "-"}Rs
                {parseFloat(Number(amount).toFixed(2))}
              </h6>
              {type === "income" ? <LuTrendingUp /> : <LuTrendingDown />}
            </div>
          </div>
        </div>
      </div>
      {notes && (
        <div className="ml-0 md:ml-16 text-xs text-gray-600 bg-purple-50 border-l-2 border-purple-300 px-3 py-2 rounded mt-1">
          <span className="font-semibold text-purple-700">📝 Note: </span>
          <span className="text-gray-700">{notes}</span>
        </div>
      )}
    </div>
  );
};

export default TransactionInfoCard;
